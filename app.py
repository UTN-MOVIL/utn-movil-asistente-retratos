# app.py — Sanic + WS/HTTP + (WebRTC via blueprint)
from __future__ import annotations

from sanic import Sanic, response
from sanic.log import logger

import os
import asyncio
import json
from pathlib import Path
from typing import Optional, List, Tuple  # (not strictly needed now, but left for typing parity)

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision

# ─────────────── Plugin registry (generic) ───────────────
from inference.core import InferencePlugin, InferenceRuntime
from inference.pose_plugin import POSE_PLUGIN
from inference.face_plugin import FACE_PLUGIN

PLUGINS: dict[str, InferencePlugin] = {
    "pose": POSE_PLUGIN,
    "face": FACE_PLUGIN,
}
RUNTIMES: dict[str, InferenceRuntime] = {}

# ─────────────── WebRTC en módulo aparte ───────────────
from connection.webrtc import build_webrtc_blueprint, TaskAdapter  # <— uses registry below

app = Sanic("MiAppHttpWebSocket")

# ─────────────── Lifecycle (generic setup/cleanup) ───────────────
@app.listener("before_server_start")
async def _setup(app, loop):
    ROOT = Path(__file__).resolve().parent
    MODEL_DIR = ROOT / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, plugin in PLUGINS.items():
        rt = InferenceRuntime(plugin, MODEL_DIR)
        RUNTIMES[name] = rt

        # resolve model path and ensure it exists
        model_path = Path(os.getenv(plugin.env_model_path_key,
                                    str(MODEL_DIR / plugin.default_model_filename)))
        plugin.ensure_model(model_path, plugin.model_urls, min_bytes=1_000_000)
        rt.model_path = model_path

        # create IMAGE landmarker
        cfg_img = plugin.make_config(model_path, mp_vision.RunningMode.IMAGE)
        rt.image_lmk = plugin.factory_cls(cfg_img).create_with_fallback()
        logger.info(f"{name} (IMAGE) inicializado.")

        # optionally create VIDEO landmarker
        use_video = False
        if plugin.use_video_env_key:
            use_video = os.getenv(plugin.use_video_env_key, "0") == "1"
        if use_video:
            cfg_vid = plugin.make_config(model_path, mp_vision.RunningMode.VIDEO)
            rt.video_lmk = plugin.factory_cls(cfg_vid).create_with_fallback()
            logger.info(f"{name} (VIDEO) inicializado.")
        else:
            logger.info(f"{name}: VIDEO desactivado; reutilizará IMAGE si aplica.")

@app.listener("after_server_stop")
async def _cleanup(app, loop):
    for rt in RUNTIMES.values():
        rt.close()
    RUNTIMES.clear()
    logger.info("Landmarkers liberados.")

# ─────────────── Generic helpers (detect + draw) ───────────────
def _make_mp_image_from_bgr(img_bgr: np.ndarray):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

async def detect_image(task: str, mp_image: mp.Image):
    rt = RUNTIMES[task]
    async with rt.lock:
        return rt.image_lmk.detect(mp_image)

async def detect_video(task: str, mp_image: mp.Image, ts_ms: int):
    rt = RUNTIMES[task]
    async with rt.lock:
        if rt.video_lmk is not None:
            return rt.video_lmk.detect_for_video(mp_image, ts_ms)
        return rt.image_lmk.detect(mp_image)  # graceful fallback

async def process_image(task: str, img_bgr: np.ndarray, return_image: bool):
    rt = RUNTIMES[task]
    plugin = rt.plugin
    mp_img = _make_mp_image_from_bgr(img_bgr)
    result = await detect_image(task, mp_img)
    if not return_image or not plugin.draw_on_bgr:
        return None, result
    frame = img_bgr.copy()
    plugin.draw_on_bgr(frame, result)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("No se pudo codificar JPEG.")
    return buf.tobytes(), result

# ─────────────── WS genérico para cualquier tarea registrada ───────────────
@app.websocket("/ws/<task>")
async def ws_generic(request, ws, task: str):
    if task not in PLUGINS:
        await ws.send(f"Tarea desconocida: {task}. Disponibles: {list(PLUGINS)}")
        await ws.close(code=1008, reason="unknown task")
        return

    print(f">>> WS/{task} conectado. Enviar binario (JPEG/PNG); 'bye' para cerrar.")
    plugin = PLUGINS[task]

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

            _, result = await process_image(task, img, return_image=False)
            payload = plugin.to_json(result, img.shape)
            await ws.send(json.dumps(payload))
        except Exception as e:
            print(f">>> ERROR en ws/{task}: {e}")
            break
    print(f">>> WS/{task} desconectado.")

# ─────────────── Registrar el Blueprint WebRTC a partir del registry ───────────────
def _make_mp_image_srgba(rgb_np: np.ndarray):
    # keep your SRGBA variant used by WebRTC, if needed
    return mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgb_np)

adapters = {}
for name, plugin in PLUGINS.items():
    def _mk_detect_image(nm):
        async def inner(mp_img): return await detect_image(nm, mp_img)
        return inner
    def _mk_detect_video(nm):
        async def inner(mp_img, ts_ms): return await detect_video(nm, mp_img, ts_ms)
        return inner
    adapters[name] = TaskAdapter(
        name=name,
        make_mp_image=_make_mp_image_srgba,
        detect_image=_mk_detect_image(name),
        detect_video=_mk_detect_video(name),
        points_from_result=plugin.points_from_result,
    )

webrtc_bp = build_webrtc_blueprint(adapters=adapters, url_prefix="")
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
        "Servidor Sanic OK. Prueba /ws, /http, /ws/<tarea> (p. ej. /ws/pose, /ws/face) o /webrtc/offer (POST signaling)."
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

# ─────────────── Main ───────────────
if __name__ == "__main__":
    # Nota: dev=True y debug=True para desarrollo; desactívalo en producción.
    app.run(host="0.0.0.0", port=8000, dev=True, debug=True)
