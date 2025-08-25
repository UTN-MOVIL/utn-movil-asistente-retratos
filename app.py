# app.py — Sanic + WS/HTTP + (WebRTC via blueprint)
from __future__ import annotations

from sanic import Sanic, response
from sanic.log import logger

import os
import asyncio
import json
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision

# ─────────────── Tus módulos propios ───────────────
# Pose (estilo "puntos_faciales" con MediaPipe Tasks)
from modules.esqueleto import (
    AppConfig as PoseAppConfig,
    LandmarkerFactory as PoseLandmarkerFactory,
    ensure_file as ensure_pose_model,
    DEFAULT_MODEL_URLS as POSE_MODEL_URLS,
    draw_pose_skeleton_bgr,
)

# Face (tu módulo existente “puntos_faciales”)
from modules.puntos_faciales import (
    AppConfig as FaceAppConfig,
    LandmarkerFactory as FaceLandmarkerFactory,
    ensure_file as ensure_face_model,
    DEFAULT_MODEL_URLS as FACE_MODEL_URLS,
    draw_landmarks_bgr as face_draw_landmarks,
)

# ─────────────── WebRTC en módulo aparte ───────────────
from connection.webrtc import build_webrtc_blueprint, TaskAdapter  # <— UPDATED

app = Sanic("MiAppHttpWebSocket")

# ─────────────── Globals (pose + face + locks) ───────────────
pose_landmarker_image: Optional[object] = None
pose_landmarker_video: Optional[object] = None
pose_lock: asyncio.Lock | None = None

face_landmarker: Optional[object] = None
face_lock: asyncio.Lock | None = None

# ─────────────── Flags/ENV necesarios aquí ───────────────
POSE_USE_VIDEO = os.getenv("POSE_USE_VIDEO", "0") == "1"

# ─────────────── Lifecycle ───────────────
@app.listener("before_server_start")
async def _setup(app, loop):
    """Inicializa modelos de Pose y Face (IMAGE y opcional VIDEO)."""
    global pose_landmarker_image, pose_landmarker_video, pose_lock
    global face_landmarker, face_lock

    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent if HERE.name == "tests" else HERE
    MODEL_DIR = ROOT / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Locks
    if pose_lock is None:
        pose_lock = asyncio.Lock()
    if face_lock is None:
        face_lock = asyncio.Lock()

    # ---- Pose (IMAGE) ----
    POSE_MODEL_PATH = Path(
        os.getenv("POSE_LANDMARKER_PATH", str(MODEL_DIR / "pose_landmarker.task"))
    )
    ensure_pose_model(POSE_MODEL_PATH, POSE_MODEL_URLS, min_bytes=1_000_000)

    pose_cfg_image = PoseAppConfig(
        model_path=POSE_MODEL_PATH,
        model_urls=list(POSE_MODEL_URLS),
        delegate_preference="gpu",                 # "gpu" | "cpu" | "auto"
        running_mode=mp_vision.RunningMode.IMAGE,  # HTTP/WS imágenes sueltas
        max_poses=1,
        min_pose_detection_confidence=0.5,
    )
    pose_landmarker_image = PoseLandmarkerFactory(pose_cfg_image).create_with_fallback()
    logger.info("PoseLandmarker (IMAGE) inicializado.")

    # ---- Pose (VIDEO) opcional para WebRTC ----
    if POSE_USE_VIDEO:
        pose_cfg_video = PoseAppConfig(
            model_path=POSE_MODEL_PATH,
            model_urls=list(POSE_MODEL_URLS),
            delegate_preference="gpu",
            running_mode=mp_vision.RunningMode.VIDEO,  # WebRTC streaming
            max_poses=1,
            min_pose_detection_confidence=0.3,         # más laxo
            min_tracking_confidence=0.2,
        )
        pose_landmarker_video = PoseLandmarkerFactory(pose_cfg_video).create_with_fallback()
        logger.info("PoseLandmarker (VIDEO) inicializado.")
    else:
        pose_landmarker_video = None
        logger.info("POSE_USE_VIDEO=0 → WebRTC usará PoseLandmarker (IMAGE).")

    # ---- Face (IMAGE) ----
    FACE_MODEL_PATH = Path(
        os.getenv("FACE_LANDMARKER_PATH", str(MODEL_DIR / "face_landmarker.task"))
    )
    ensure_face_model(FACE_MODEL_PATH, FACE_MODEL_URLS, min_bytes=1_000_000)

    face_cfg = FaceAppConfig(
        model_path=FACE_MODEL_PATH,
        model_urls=list(FACE_MODEL_URLS),
        delegate_preference="gpu",
        running_mode=mp_vision.RunningMode.IMAGE,
        max_faces=1,
        min_face_detection_confidence=0.5,
    )
    face_landmarker = FaceLandmarkerFactory(face_cfg).create_with_fallback()
    logger.info("FaceLandmarker (IMAGE) inicializado.")

@app.listener("after_server_stop")
async def _cleanup(app, loop):
    """Libera los recursos de los landmarkers."""
    global pose_landmarker_image, pose_landmarker_video, face_landmarker

    try:
        if pose_landmarker_image and hasattr(pose_landmarker_image, "close"):
            pose_landmarker_image.close()
    except Exception:
        pass
    pose_landmarker_image = None

    try:
        if pose_landmarker_video and hasattr(pose_landmarker_video, "close"):
            pose_landmarker_video.close()
    except Exception:
        pass
    pose_landmarker_video = None

    try:
        if face_landmarker and hasattr(face_landmarker, "close"):
            face_landmarker.close()
    except Exception:
        pass
    face_landmarker = None

    logger.info("Pose/Face Landmarkers liberados.")

# ─────────────── Serializadores / Procesamiento (HTTP/WS) ───────────────
def _results_pose_to_json(result, img_shape):
    h, w = img_shape[:2]
    if not result or not getattr(result, "pose_landmarks", None):
        return {"poses": [], "image_size": {"w": w, "h": h}, "num_poses": 0}

    poses = []
    for lms in result.pose_landmarks:
        poses.append(
            [
                {
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(getattr(lm, "visibility", 0.0)),
                    "px": float(lm.x * w),
                    "py": float(lm.y * h),
                }
                for lm in lms
            ]
        )
    return {"poses": poses, "image_size": {"w": w, "h": h}, "num_poses": len(poses)}

async def _process_pose(img_bgr: np.ndarray, return_image: bool):
    """Corre Pose (IMAGE) y opcionalmente dibuja, devolviendo JPEG bytes."""
    global pose_landmarker_image, pose_lock
    if pose_landmarker_image is None or pose_lock is None:
        raise RuntimeError("PoseLandmarker (IMAGE) no está inicializado.")

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    async with pose_lock:
        result = pose_landmarker_image.detect(mp_image)

    if not return_image:
        return None, result

    frame = img_bgr.copy()
    draw_pose_skeleton_bgr(frame, result)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("No se pudo codificar JPEG.")
    return buf.tobytes(), result

def _results_face_to_json(result, img_shape):
    h, w = img_shape[:2]
    if not result or not getattr(result, "face_landmarks", None):
        return {"faces": [], "image_size": {"w": w, "h": h}, "num_faces": 0}

    faces = []
    for landmarks in result.face_landmarks:
        faces.append(
            [
                {
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "px": float(lm.x * w),
                    "py": float(lm.y * h),
                }
                for lm in landmarks
            ]
        )
    return {"faces": faces, "image_size": {"w": w, "h": h}, "num_faces": len(faces)}

async def _process_face(img_bgr: np.ndarray, return_image: bool):
    """Corre Face (IMAGE) y opcionalmente dibuja, devolviendo JPEG bytes."""
    global face_landmarker, face_lock
    if face_landmarker is None or face_lock is None:
        raise RuntimeError("FaceLandmarker no está inicializado.")

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    async with face_lock:
        result = face_landmarker.detect(mp_image)

    if not return_image:
        return None, result

    frame = img_bgr.copy()
    face_draw_landmarks(frame, result)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("No se pudo codificar JPEG.")
    return buf.tobytes(), result

def _poses_px_from_result(result, img_shape) -> Tuple[int, int, List[List[Tuple[int, int]]]]:
    """Convierte landmarks normalizados → píxeles absolutos."""
    h, w = img_shape[:2]
    poses_px: List[List[Tuple[int, int]]] = []
    if not result or not getattr(result, "pose_landmarks", None):
        return w, h, poses_px
    for lms in result.pose_landmarks:
        pts: List[Tuple[int, int]] = []
        for lm in lms:
            x = int(round(lm.x * w))
            y = int(round(lm.y * h))
            x = 0 if x < 0 else (w - 1 if x >= w else x)
            y = 0 if y < 0 else (h - 1 if y >= h else y)
            pts.append((x, y))
        poses_px.append(pts)
    return w, h, poses_px

# ───────── Face → píxeles y wrappers (para WebRTC) ─────────
def _faces_px_from_result(result, img_shape) -> Tuple[int, int, List[List[Tuple[int, int]]]]:
    """Convierte landmarks faciales normalizados → píxeles absolutos."""
    h, w = img_shape[:2]
    faces_px: List[List[Tuple[int, int]]] = []
    if not result or not getattr(result, "face_landmarks", None):
        return w, h, faces_px
    for lms in result.face_landmarks:
        pts: List[Tuple[int, int]] = []
        for lm in lms:
            x = int(round(lm.x * w))
            y = int(round(lm.y * h))
            x = 0 if x < 0 else (w - 1 if x >= w else x)
            y = 0 if y < 0 else (h - 1 if y >= h else y)
            pts.append((x, y))
        faces_px.append(pts)
    return w, h, faces_px

def _make_mp_image(rgb_np: np.ndarray):
    return mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgb_np)

async def _detect_pose_image(mp_image: mp.Image):
    global pose_landmarker_image, pose_lock
    if pose_landmarker_image is None or pose_lock is None:
        raise RuntimeError("PoseLandmarker (IMAGE) no está inicializado.")
    async with pose_lock:
        return pose_landmarker_image.detect(mp_image)

async def _detect_pose_video(mp_image: mp.Image, ts_ms: int):
    global pose_landmarker_video, pose_landmarker_image, pose_lock
    if pose_lock is None:
        raise RuntimeError("pose_lock no está inicializado.")
    async with pose_lock:
        if pose_landmarker_video is not None:
            return pose_landmarker_video.detect_for_video(mp_image, ts_ms)
        if pose_landmarker_image is None:
            raise RuntimeError("No hay landmarker de pose inicializado.")
        return pose_landmarker_image.detect(mp_image)

async def _detect_face_image(mp_image: mp.Image):
    """Face en modo IMAGE (usado también en WebRTC)."""
    global face_landmarker, face_lock
    if face_landmarker is None or face_lock is None:
        raise RuntimeError("FaceLandmarker no está inicializado.")
    async with face_lock:
        return face_landmarker.detect(mp_image)

async def _detect_face_video(mp_image: mp.Image, ts_ms: int):
    """Wrapper VIDEO para Face que delega a IMAGE."""
    return await _detect_face_image(mp_image)

# ───────── Registrar el Blueprint WebRTC (dos tareas: pose + face) ─────────
webrtc_bp = build_webrtc_blueprint(
    adapters={
        "pose": TaskAdapter(
            name="pose",
            make_mp_image=_make_mp_image,
            detect_image=_detect_pose_image,
            detect_video=_detect_pose_video,
            points_from_result=_poses_px_from_result,
        ),
        "face": TaskAdapter(
            name="face",
            make_mp_image=_make_mp_image,
            detect_image=_detect_face_image,
            detect_video=_detect_face_video,
            points_from_result=_faces_px_from_result,
        ),
    },
    url_prefix="",
)
app.blueprint(webrtc_bp)

# ─────────────── Endpoints HTTP/WS (no WebRTC) ───────────────
@app.route("/http", methods=["GET", "POST"])
async def http_handler(request):
    """Echo simple por HTTP (GET/POST)."""
    if request.method == "GET":
        return response.text("Hola desde /http (GET).")
    data_recibida = (
        request.json if request.json else request.form if request.form else request.body
    )
    return response.text(f"Datos recibidos vía HTTP (POST): {data_recibida}")

@app.route("/", methods=["GET"])
async def root_handler(request):
    return response.text(
        "Servidor Sanic OK. Prueba /ws, /http, /ws/pose, /ws/face o /webrtc/offer (POST signaling)."
    )

@app.websocket("/ws")
async def websocket_handler(request, ws):
    """WS echo de texto/binario."""
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

@app.websocket("/ws/pose")
async def ws_pose(request, ws):
    """Envía imagen binaria (JPEG/PNG); responde JSON de pose."""
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

            arr = np.frombuffer(msg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                await ws.send("No se pudo decodificar la imagen. Usa JPEG/PNG.")
                continue

            _, result = await _process_pose(img, return_image=False)
            payload = _results_pose_to_json(result, img.shape)
            await ws.send(json.dumps(payload))  # JSON directo
        except Exception as e:
            print(f">>> ERROR en ws/pose: {e}")
            break
    print(">>> WS/pose desconectado.")

@app.websocket("/ws/face")
async def ws_face(request, ws):
    """Envía imagen binaria (JPEG/PNG); responde JSON de face."""
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

# ─────────────── Main ───────────────
if __name__ == "__main__":
    # Nota: dev=True y debug=True para desarrollo; desactívalo en producción.
    app.run(host="0.0.0.0", port=8000, dev=True, debug=True)
