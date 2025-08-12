# app.py — Sanic + WebRTC + WS + HTTP (Pose + Face)
from __future__ import annotations

from sanic import Sanic, response
from sanic.exceptions import InvalidUsage
from sanic.log import logger

import os
import time
import asyncio
import base64
import json  # stdlib json (keep this name free!)
import struct
from pathlib import Path
from typing import Dict, Set, Optional, List, Tuple

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision

# ─────────────── Tus módulos propios ───────────────
# Pose (estilo "puntos_faciales" con MediaPipe Tasks)
from modulos.esqueleto import (
    AppConfig as PoseAppConfig,
    LandmarkerFactory as PoseLandmarkerFactory,
    ensure_file as ensure_pose_model,
    DEFAULT_MODEL_URLS as POSE_MODEL_URLS,
    draw_pose_skeleton_bgr,
)

# Face (tu módulo existente “puntos_faciales”)
from modulos.puntos_faciales import (
    AppConfig as FaceAppConfig,
    LandmarkerFactory as FaceLandmarkerFactory,
    ensure_file as ensure_face_model,
    DEFAULT_MODEL_URLS as FACE_MODEL_URLS,
    draw_landmarks_bgr as face_draw_landmarks,
)

# ─────────────── WebRTC (aiortc) ───────────────
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceServer,
    RTCConfiguration,
    MediaStreamTrack,
)
from aiortc.contrib.media import MediaRelay
import av  # PyAV frames

app = Sanic("MiAppHttpWebSocket")

# ─────────────── Globals (pose + face + locks) ───────────────
# Usamos dos instancias de Pose: IMAGE para HTTP/WS y VIDEO para WebRTC
pose_landmarker_image: Optional[object] = None
pose_landmarker_video: Optional[object] = None
pose_lock: asyncio.Lock | None = None  # serialize MediaPipe access

face_landmarker = None
face_lock: asyncio.Lock | None = None

# WebRTC globals
pcs: Set[RTCPeerConnection] = set()
relay = MediaRelay()
results_dc_by_pc: Dict[RTCPeerConnection, object] = {}

WEBRTC_ANNOTATE = os.getenv("WEBRTC_ANNOTATE", "0") == "1"
# Si pones WEBRTC_JSON_RESULTS=1, enviará JSON en lugar del binario optimizado
WEBRTC_JSON_RESULTS = os.getenv("WEBRTC_JSON_RESULTS", "0") == "1"


# ─────────────── Lifecycle ───────────────
@app.listener("before_server_start")
async def _setup(app, loop):
    global pose_landmarker_image, pose_landmarker_video, pose_lock
    global face_landmarker, face_lock

    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent if HERE.name == "tests" else HERE
    MODEL_DIR = ROOT / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Pose (IMAGE) para HTTP/WS ----
    if pose_lock is None:
        pose_lock = asyncio.Lock()

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

    # ---- Pose (VIDEO) para WebRTC ----
    pose_cfg_video = PoseAppConfig(
        model_path=POSE_MODEL_PATH,
        model_urls=list(POSE_MODEL_URLS),
        delegate_preference="gpu",
        running_mode=mp_vision.RunningMode.VIDEO,  # WebRTC streaming
        max_poses=1,
        min_pose_detection_confidence=0.5,
    )
    pose_landmarker_video = PoseLandmarkerFactory(pose_cfg_video).create_with_fallback()
    logger.info("PoseLandmarker (VIDEO) inicializado.")

    # ---- Face Landmarker (IMAGE) ----
    if face_lock is None:
        face_lock = asyncio.Lock()

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
    logger.info("FaceLandmarker inicializado.")


@app.listener("after_server_stop")
async def _cleanup(app, loop):
    global pose_landmarker_image, pose_landmarker_video, face_landmarker

    # Pose (IMAGE)
    try:
        if pose_landmarker_image and hasattr(pose_landmarker_image, "close"):
            pose_landmarker_image.close()
    except Exception:
        pass
    pose_landmarker_image = None

    # Pose (VIDEO)
    try:
        if pose_landmarker_video and hasattr(pose_landmarker_video, "close"):
            pose_landmarker_video.close()
    except Exception:
        pass
    pose_landmarker_video = None
    logger.info("PoseLandmarkers liberados.")

    # Face
    try:
        if face_landmarker and hasattr(face_landmarker, "close"):
            face_landmarker.close()
    except Exception:
        pass
    face_landmarker = None
    logger.info("FaceLandmarker liberado.")

    # WebRTC peers
    for pc in list(pcs):
        try:
            await pc.close()
        except Exception:
            pass
    pcs.clear()
    results_dc_by_pc.clear()
    logger.info("RTCPeerConnections cerrados.")


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


# ---- Pose JSON serializer (Tasks) ----
def _results_pose_to_json(result, img_shape):
    """
    Convierte PoseLandmarkerResult a JSON.
    Devuelve todas las poses como listas de puntos (x,y,z, px,py, visibility).
    """
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
    """Ejecuta Pose Landmarker (modo IMAGE) bajo lock; opcionalmente devuelve JPEG anotado."""
    global pose_landmarker_image, pose_lock
    if pose_landmarker_image is None or pose_lock is None:
        raise RuntimeError("PoseLandmarker (IMAGE) no está inicializado.")

    # MediaPipe Tasks espera RGB
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    async with pose_lock:
        result = pose_landmarker_image.detect(mp_image)  # RunningMode.IMAGE

    if not return_image:
        return None, result

    frame = img_bgr.copy()
    draw_pose_skeleton_bgr(frame, result)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("No se pudo codificar JPEG.")
    return buf.tobytes(), result


# ---- Face JSON serializer ----
def _results_face_to_json(result, img_shape):
    """Convierte FaceLandmarkerResult a JSON simple."""
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
    """Ejecuta Face Landmarker (modo IMAGE) bajo lock; opcionalmente devuelve JPEG anotado."""
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


# ─────────────── Binary packer for WebRTC results (matches Flutter client) ───────────────
# Python example
def pack_pose_frame(image_w: int, image_h: int, poses: List[List[Tuple[int, int]]]) -> bytes:
    """Empaqueta poses en binario compacto:
    'P''O', u8 version(0), u8 numPoses, u16 w, u16 h, [por pose: u8 numPts, (u16 x, u16 y) * numPts]
    """
    out = bytearray()
    out += b"PO"                 # magic
    out += bytes([0])            # version
    out += bytes([len(poses)])   # num poses
    out += struct.pack("<HH", image_w, image_h)

    for pts in poses:
      out += bytes([len(pts)])
      for (x, y) in pts:
        # clamp to image bounds; uint16 pixel coords
        out += struct.pack("<HH", max(0, min(65535, x)), max(0, min(65535, y)))
    return bytes(out)


def _poses_px_from_result(result, img_shape) -> Tuple[int, int, List[List[Tuple[int, int]]]]:
    """Extrae listas de (px,py) como enteros desde PoseLandmarkerResult."""
    h, w = img_shape[:2]
    poses_px: List[List[Tuple[int, int]]] = []
    if not result or not getattr(result, "pose_landmarks", None):
        return w, h, poses_px

    for lms in result.pose_landmarks:
        pts: List[Tuple[int, int]] = []
        for lm in lms:
            x = int(round(lm.x * w))
            y = int(round(lm.y * h))
            # Limitar a la imagen
            x = 0 if x < 0 else (w - 1 if x >= w else x)
            y = 0 if y < 0 else (h - 1 if y >= h else y)
            pts.append((x, y))
        poses_px.append(pts)
    return w, h, poses_px


# ─────────────── HTTP: /pose ───────────────
@app.post("/pose")
async def pose_handler(request):
    mode = (request.args.get("mode") or "json").lower()
    if mode not in {"json", "image"}:
        raise InvalidUsage("Parámetro mode debe ser 'json' o 'image'.")

    img = _decode_image_from_request(request)
    want_image = (mode == "image")
    jpeg_bytes, result = await _process_pose(img, return_image=want_image)

    if want_image:
        return response.raw(jpeg_bytes, content_type="image/jpeg")
    else:
        payload = _results_pose_to_json(result, img.shape)
        return response.json(payload)


# ─────────────── HTTP: /face ───────────────
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
@app.websocket("/ws")
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
# Enviar BINARY (JPEG/PNG) -> recibir JSON (texto)
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

            arr = np.frombuffer(msg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                await ws.send("No se pudo decodificar la imagen. Usa JPEG/PNG.")
                continue

            _, result = await _process_pose(img, return_image=False)
            payload = _results_pose_to_json(result, img.shape)
            await ws.send(json.dumps(payload))

        except Exception as e:
            print(f">>> ERROR en ws/pose: {e}")
            break
    print(">>> WS/pose desconectado.")


# ─────────────── WebSocket: face stream (/ws/face) ───────────────
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


# ─────────────── WebRTC: signaling + transform track ───────────────
class PoseTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track: MediaStreamTrack, pc: RTCPeerConnection):
        super().__init__()
        self._track = relay.subscribe(track)
        self._pc = pc

    async def recv(self) -> av.VideoFrame:
        frame: av.VideoFrame = await self._track.recv()
        img_bgr = frame.to_ndarray(format="bgr24")

        result = None
        try:
            # latest-only: si hay una inferencia en curso, no bloquees: pasa el frame
            if pose_lock is not None and pose_lock.locked():
                return frame

            # MediaPipe Tasks espera RGB
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(time.time() * 1000)

            async with pose_lock:
                if pose_landmarker_video is not None:
                    result = pose_landmarker_video.detect_for_video(mp_image, ts_ms)
                else:
                    # fallback (no debería ocurrir)
                    result = pose_landmarker_image.detect(mp_image)

            # Envío de resultados por DataChannel: binario (rápido) o JSON (fallback)
            dc = results_dc_by_pc.get(self._pc)
            if dc and getattr(dc, "readyState", "") == "open":
                # Evita congestión: si hay demasiado en búfer, salta este envío
                if getattr(dc, "bufferedAmount", 0) < 64_000:
                    if WEBRTC_JSON_RESULTS:
                        payload = _results_pose_to_json(result, img_bgr.shape)
                        payload["ts_ms"] = ts_ms
                        dc.send(json.dumps(payload))
                    else:
                        w, h, poses_px = _poses_px_from_result(result, img_bgr.shape)
                        packet = pack_pose_frame(w, h, poses_px)
                        dc.send(packet)

        except Exception as e:
            logger.warning("Pose processing error: %s", e)

        if not WEBRTC_ANNOTATE:
            return frame

        # Anotar el frame usando el resultado ya calculado (si existe)
        overlay = img_bgr.copy()
        try:
            if result is not None:
                draw_pose_skeleton_bgr(overlay, result)
        except Exception as e:
            logger.warning("Annotation error: %s", e)
            overlay = img_bgr

        out = av.VideoFrame.from_ndarray(overlay, format="bgr24")
        out.pts, out.time_base = frame.pts, frame.time_base
        return out


def _rtc_configuration() -> RTCConfiguration:
    stun_url = os.getenv("STUN_URL", "stun:stun.l.google.com:19302")
    ice_servers = [RTCIceServer(urls=stun_url)]
    turn_url = os.getenv("TURN_URL")
    if turn_url:
        ice_servers.append(
            RTCIceServer(
                urls=turn_url,
                username=os.getenv("TURN_USERNAME"),
                credential=os.getenv("TURN_PASSWORD"),
            )
        )
    return RTCConfiguration(iceServers=ice_servers)


@app.post("/webrtc/offer")
async def webrtc_offer(request):
    """Signaling endpoint: accepts SDP offer, returns SDP answer.
    Client should add a video track and (optionally) create a datachannel named 'results'.
    """
    params = request.json or {}
    if "sdp" not in params or "type" not in params:
        raise InvalidUsage("Body JSON must contain 'sdp' and 'type'.")

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection(configuration=_rtc_configuration())
    pcs.add(pc)
    logger.info("Created PC %s", id(pc))

    # Create our results datachannel proactively (client can also create one)
    try:
        # unordered + maxRetransmits=0 => drop late packets (latest-only)
        dc = pc.createDataChannel("results", ordered=False, maxRetransmits=0)
        results_dc_by_pc[pc] = dc

        @dc.on("open")
        def _on_open():
            logger.info("DataChannel 'results' open on PC %s", id(pc))

        @dc.on("close")
        def _on_close():
            logger.info("DataChannel 'results' closed on PC %s", id(pc))
    except Exception:
        pass

    @pc.on("datachannel")
    def on_dc(channel):
        # If client creates it, keep the reference (may be reliable; we still use latest-only logic)
        results_dc_by_pc[pc] = channel
        logger.info("Data channel '%s' open on PC %s", channel.label, id(pc))

    @pc.on("track")
    def on_track(track):
        logger.info("Track %s received (kind=%s) on PC %s", track.id, track.kind, id(pc))
        if track.kind == "video":
            pc.addTrack(PoseTransformTrack(track, pc))

    @pc.on("connectionstatechange")
    async def on_state_change():
        state = pc.connectionState
        logger.info("PC %s state: %s", id(pc), state)
        if state in ("failed", "closed"):
            try:
                await pc.close()
            finally:
                pcs.discard(pc)
                results_dc_by_pc.pop(pc, None)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return response.json(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


# ─────────────── Simple HTTP endpoints ───────────────
@app.route("/http", methods=["GET", "POST"])
async def http_handler(request):
    if request.method == "GET":
        return response.text("Hola desde /http (GET).")
    data_recibida = (
        request.json if request.json else request.form if request.form else request.body
    )
    return response.text(f"Datos recibidos vía HTTP (POST): {data_recibida}")


@app.route("/", methods=["GET"])
async def root_handler(request):
    return response.text(
        "Servidor Sanic OK. Prueba /ws, /http, /pose, /face, /ws/pose, /ws/face o /webrtc/offer (POST signaling)."
    )


if __name__ == "__main__":
    # Nota: dev=True y debug=True para desarrollo; desactívalo en prod.
    app.run(host="0.0.0.0", port=8000, dev=True, debug=True)
