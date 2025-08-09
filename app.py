# app.py
from sanic import Sanic, response
from sanic.exceptions import InvalidUsage
from sanic.log import logger

import asyncio
import base64
import json  # stdlib json (keep this name free!)
import numpy as np
import cv2

# Your own module
from modulos.esqueleto import PoseTracker, PoseConfig

app = Sanic("MiAppHttpWebSocket")

# ─────────────── Globals (tracker + lock) ───────────────
pose_tracker: PoseTracker | None = None
pose_lock: asyncio.Lock | None = None  # serialize MediaPipe access

@app.listener("before_server_start")
async def _setup(app, loop):
    global pose_tracker, pose_lock
    pose_lock = asyncio.Lock()
    cfg = PoseConfig(
        flip_display=False,
        draw_landmarks=True,
        show_fps=False,
        model_complexity=1,
        min_det_conf=0.5,
        min_track_conf=0.5,
    )
    pose_tracker = PoseTracker(cfg)
    pose_tracker.__enter__()  # open once
    logger.info("PoseTracker inicializado.")

@app.listener("after_server_stop")
async def _cleanup(app, loop):
    global pose_tracker
    if pose_tracker:
        pose_tracker.__exit__(None, None, None)
        pose_tracker = None
        logger.info("PoseTracker liberado.")

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

def _results_to_json(results, img_shape):
    """Flatten MediaPipe landmarks to a simple JSON-friendly dict."""
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
    """Run the MediaPipe pipeline under a lock; optionally draw and return JPEG."""
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
        payload = _results_to_json(results, img.shape)
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
            payload = _results_to_json(results, img.shape)
            await ws.send(json.dumps(payload))  # ✅ stdlib json

        except Exception as e:
            print(f">>> ERROR en ws/pose: {e}")
            break
    print(">>> WS/pose desconectado.")

# ─────────────── Simple HTTP endpoints ───────────────
@app.route('/http', methods=['GET', 'POST'])
async def http_handler(request):
    if request.method == 'GET':
        return response.text("Hola desde /http (GET).")
    data_recibida = request.json if request.json else request.form if request.form else request.body
    return response.text(f"Datos recibidos vía HTTP (POST): {data_recibida}")

@app.route('/', methods=['GET'])
async def root_handler(request):
    return response.text("Servidor Sanic OK. Prueba /ws, /http, /pose o /ws/pose.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, dev=True, debug=True)
