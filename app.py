# app.py
from sanic import Sanic, response
from sanic.exceptions import InvalidUsage
from sanic.log import logger

import os
import asyncio
import base64
import json  # stdlib json (keep this name free!)
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision

# Tus módulos propios
from modulos.esqueleto import PoseTracker, PoseConfig
from modulos.puntos_faciales import (
    AppConfig as FaceAppConfig,
    LandmarkerFactory as FaceLandmarkerFactory,
    ensure_file as ensure_face_model,
    DEFAULT_MODEL_URLS as FACE_MODEL_URLS,
    draw_landmarks_bgr as face_draw_landmarks,
)

app = Sanic("MiAppHttpWebSocket")

# ─────────────── Globals (pose + face + locks) ───────────────
pose_tracker: PoseTracker | None = None
pose_lock: asyncio.Lock | None = None  # serialize MediaPipe access

face_landmarker = None
face_lock: asyncio.Lock | None = None  # serialize MediaPipe access


# ─────────────── Lifecycle ───────────────
@app.listener("before_server_start")
async def _setup(app, loop):
    global pose_tracker, pose_lock, face_landmarker, face_lock

    # ---- Pose ----
    pose_lock = asyncio.Lock()
    pose_cfg = PoseConfig(
        flip_display=False,
        draw_landmarks=True,
        show_fps=False,
        model_complexity=1,
        min_det_conf=0.5,
        min_track_conf=0.5,
    )
    pose_tracker = PoseTracker(pose_cfg)
    pose_tracker.__enter__()  # open once
    logger.info("PoseTracker inicializado.")

    # ---- Face Landmarker (MediaPipe Tasks) ----
    if face_lock is None:
        face_lock = asyncio.Lock()

    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent if HERE.name == "tests" else HERE
    MODEL_DIR = ROOT / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = Path(os.getenv("FACE_LANDMARKER_PATH", str(MODEL_DIR / "face_landmarker.task")))

    # Asegura que el .task exista (descarga si falta)
    ensure_face_model(MODEL_PATH, FACE_MODEL_URLS, min_bytes=1_000_000)

    face_cfg = FaceAppConfig(
        model_path=MODEL_PATH,
        model_urls=list(FACE_MODEL_URLS),
        delegate_preference="gpu",               # "gpu" | "cpu" | "auto"
        running_mode=mp_vision.RunningMode.IMAGE,
        max_faces=1,
        min_face_detection_confidence=0.5,
    )
    face_landmarker = FaceLandmarkerFactory(face_cfg).create_with_fallback()
    logger.info("FaceLandmarker inicializado.")


@app.listener("after_server_stop")
async def _cleanup(app, loop):
    global pose_tracker, face_landmarker
    if pose_tracker:
        pose_tracker.__exit__(None, None, None)
        pose_tracker = None
        logger.info("PoseTracker liberado.")

    # FaceLandmarker no requiere cierre explícito, pero por si acaso:
    try:
        if face_landmarker and hasattr(face_landmarker, "close"):
            face_landmarker.close()
    except Exception:
        pass
    face_landmarker = None
    logger.info("FaceLandmarker liberado.")


# ─────────────── Helpers ───────────────
def _decode_image_from_request(request) -> np.ndarray:
    """
    Supports:
    - multipart/form-data with field 'image'
    - application/json with { "image_b64": "<...>" }
    - application/octet-stream (raw bytes)
    Returns BGR np.ndarray; raises InvalidUsage otherwise.
    """
    # 1) multipart
    if request.files and "image" in request.files:
        body = request.files["image"][0].body
    # 2) json base64
    elif request.json and "image_b64" in request.json:
        try:
            body = base64.b64decode(request.json["image_b64"])
        except Exception:
            raise InvalidUsage("image_b64 no es base64 válido")
    # 3) raw binary
    elif request.body:
        body = request.body
    else:
        raise InvalidUsage("Debe enviar imagen por multipart (image), JSON (image_b64) o binario.")

    arr = np.frombuffer(body, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidUsage("No se pudo decodificar la imagen. Use JPEG/PNG válidos.")
    return img  # BGR


# ---- Pose JSON serializer ----
def _results_pose_to_json(results, img_shape):
    """Flatten MediaPipe landmarks (pose) to a simple JSON-friendly dict."""
    h, w = img_shape[:2]
    if not results or not getattr(results, "pose_landmarks", None):
        return {"landmarks": [], "image_size": {"w": w, "h": h}}
    lms = results.pose_landmarks.landmark
    out = [
        {
            "x": float(pt.x),
            "y": float(pt.y),
            "z": float(pt.z),
            "visibility": float(pt.visibility),
            "px": float(pt.x * w),
            "py": float(pt.y * h),
        }
        for pt in lms
    ]
    return {"landmarks": out, "image_size": {"w": w, "h": h}}


async def _process_pose(img_bgr: np.ndarray, return_image: bool):
    """Run the Pose pipeline under a lock; optionally draw and return JPEG."""
    global pose_tracker, pose_lock
    if pose_tracker is None or pose_lock is None:
        raise RuntimeError("PoseTracker no está inicializado.")

    async with pose_lock:
        results = pose_tracker.process(img_bgr)

        if return_image:
            frame = img_bgr.copy()
            pose_tracker.draw(frame, results)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError("No se pudo codificar JPEG.")
            return buf.tobytes(), results
        else:
            return None, results


# ---- Face JSON serializer ----
def _results_face_to_json(result, img_shape):
    """Convierte FaceLandmarkerResult a JSON simple."""
    h, w = img_shape[:2]
    if not result or not getattr(result, "face_landmarks", None):
        return {"faces": [], "image_size": {"w": w, "h": h}, "num_faces": 0}

    faces = []
    for landmarks in result.face_landmarks:  # lista por cada rostro
        faces.append([
            {
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "px": float(lm.x * w),
                "py": float(lm.y * h),
            } for lm in landmarks
        ])
    return {"faces": faces, "image_size": {"w": w, "h": h}, "num_faces": len(faces)}


async def _process_face(img_bgr: np.ndarray, return_image: bool):
    """Ejecuta Face Landmarker (modo IMAGE) bajo lock; opcionalmente devuelve JPEG anotado."""
    global face_landmarker, face_lock
    if face_landmarker is None or face_lock is None:
        raise RuntimeError("FaceLandmarker no está inicializado.")

    # MediaPipe Tasks espera RGB
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    async with face_lock:
        result = face_landmarker.detect(mp_image)  # RunningMode.IMAGE

    if not return_image:
        return None, result

    frame = img_bgr.copy()
    face_draw_landmarks(frame, result)  # reutiliza función del módulo
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("No se pudo codificar JPEG.")
    return buf.tobytes(), result


# ─────────────── HTTP: /pose ───────────────
# POST /pose?mode=json   -> JSON with landmarks
# POST /pose?mode=image  -> annotated JPEG
@app.post("/pose")
async def pose_handler(request):
    mode = (request.args.get("mode") or "json").lower()
    if mode not in {"json", "image"}:
        raise InvalidUsage("Parámetro mode debe ser 'json' o 'image'.")

    img = _decode_image_from_request(request)
    want_image = (mode == "image")
    jpeg_bytes, results = await _process_pose(img, return_image=want_image)

    if want_image:
        return response.raw(jpeg_bytes, content_type="image/jpeg")
    else:
        payload = _results_pose_to_json(results, img.shape)
        return response.json(payload)


# ─────────────── HTTP: /face ───────────────
# POST /face?mode=json   -> JSON con landmarks faciales
# POST /face?mode=image  -> JPEG anotado con landmarks
@app.post("/face")
async def face_handler(request):
    mode = (request.args.get("mode") or "json").lower()
    if mode not in {"json", "image"}:
        raise InvalidUsage("Parámetro mode debe ser 'json' o 'image'.")

    img = _decode_image_from_request(request)
    want_image = (mode == "image")
    jpeg_bytes, result = await _process_face(img, return_image=want_image)

    if want_image:
        return response.raw(jpeg_bytes, content_type="image/jpeg")
    else:
        payload = _results_face_to_json(result, img.shape)
        return response.json(payload)


# ─────────────── WebSocket: echo (/ws) ───────────────
@app.websocket('/ws')
async def websocket_handler(request, ws):
    print(">>> Conexión WebSocket establecida. Esperando mensajes...")
    while True:
        try:
            data = await ws.recv()
            if not data:
                break
            await ws.send(f"Recibido vía WebSocket: {data}")
        except Exception as e:
            print(f">>> ERROR en el manejador WebSocket: {e}")
            break
    print(">>> Manejador WebSocket finalizado para esta conexión.")


# ─────────────── WebSocket: pose stream (/ws/pose) ───────────────
# Send BINARY (JPEG/PNG) -> receive JSON (text)
@app.websocket("/ws/pose")
async def ws_pose(request, ws):
    print(">>> WS/pose conectado. Enviar binario (JPEG/PNG); 'bye' para cerrar.")
    while True:
        try:
            msg = await ws.recv()
            if isinstance(msg, str):
                if msg.lower().strip() in {"bye", "close"}:
                    await ws.send("closing")
                    await ws.close(code=1000, reason="bye")
                    break
                await ws.send("Envía imagen binaria (JPEG/PNG) o 'bye' para cerrar.")
                continue

            # Expecting binary image
            arr = np.frombuffer(msg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                await ws.send("No se pudo decodificar la imagen. Usa JPEG/PNG.")
                continue

            _, results = await _process_pose(img, return_image=False)
            payload = _results_pose_to_json(results, img.shape)
            await ws.send(json.dumps(payload))  # ✅ stdlib json

        except Exception as e:
            print(f">>> ERROR en ws/pose: {e}")
            break
    print(">>> WS/pose desconectado.")


# ─────────────── WebSocket: face stream (/ws/face) ───────────────
# Enviar BINARY (JPEG/PNG) -> recibir JSON (texto)
@app.websocket("/ws/face")
async def ws_face(request, ws):
    print(">>> WS/face conectado. Enviar binario (JPEG/PNG); 'bye' para cerrar.")
    while True:
        try:
            msg = await ws.recv()
            if isinstance(msg, str):
                if msg.lower().strip() in {"bye", "close"}:
                    await ws.send("closing")
                    await ws.close(code=1000, reason="bye")
                    break
                await ws.send("Envía imagen binaria (JPEG/PNG) o 'bye' para cerrar.")
                continue

            arr = np.frombuffer(msg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                await ws.send("No se pudo decodificar la imagen. Usa JPEG/PNG.")
                continue

            _, result = await _process_face(img, return_image=False)
            payload = _results_face_to_json(result, img.shape)
            await ws.send(json.dumps(payload))
        except Exception as e:
            print(f">>> ERROR en ws/face: {e}")
            break
    print(">>> WS/face desconectado.")


# ─────────────── Simple HTTP endpoints ───────────────
@app.route('/http', methods=['GET', 'POST'])
async def http_handler(request):
    if request.method == 'GET':
        return response.text("Hola desde /http (GET).")
    data_recibida = request.json if request.json else request.form if request.form else request.body
    return response.text(f"Datos recibidos vía HTTP (POST): {data_recibida}")


@app.route('/', methods=['GET'])
async def root_handler(request):
    return response.text("Servidor Sanic OK. Prueba /ws, /http, /pose, /face, /ws/pose o /ws/face.")


if __name__ == "__main__":
    # Nota: dev=True y debug=True para desarrollo; desactívalo en prod.
    app.run(host="0.0.0.0", port=8000, dev=True, debug=True)
