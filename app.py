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
pose_landmarker_image: Optional[object] = None
pose_landmarker_video: Optional[object] = None
pose_lock: asyncio.Lock | None = None

face_landmarker = None
face_lock: asyncio.Lock | None = None

# WebRTC globals
pcs: Set[RTCPeerConnection] = set()
relay = MediaRelay()
results_dc_by_pc: Dict[RTCPeerConnection, object] = {}
ctrl_dc_by_pc: Dict[RTCPeerConnection, object] = {}
need_keyframe_by_pc: Dict[RTCPeerConnection, bool] = {}

# per-PC sequence for PD packets (detect loss/reorder on client)
results_seq_by_pc: Dict[RTCPeerConnection, int] = {}

WEBRTC_ANNOTATE = os.getenv("WEBRTC_ANNOTATE", "0") == "1"
WEBRTC_JSON_RESULTS = os.getenv("WEBRTC_JSON_RESULTS", "0") == "1"  # default 0 → binary PO/PD

# ─────────────── Behaviour toggles (via env) ───────────────
POSE_USE_VIDEO = os.getenv("POSE_USE_VIDEO", "0") == "1"  # default IMAGE for WebRTC path
ABSOLUTE_INTERVAL_MS = int(os.getenv("ABSOLUTE_INTERVAL_MS", "0"))    # default off
IDLE_TO_FORCE_KF_MS = int(os.getenv("IDLE_TO_FORCE_KF_MS", "500"))
FRAME_GAP_WARN_MS = int(os.getenv("FRAME_GAP_WARN_MS", "180"))

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

    # ---- Pose para WebRTC (VIDEO o IMAGE según env) ----
    if POSE_USE_VIDEO:
        pose_cfg_video = PoseAppConfig(
            model_path=POSE_MODEL_PATH,
            model_urls=list(POSE_MODEL_URLS),
            delegate_preference="gpu",
            running_mode=mp_vision.RunningMode.VIDEO,  # WebRTC streaming
            max_poses=1,
            min_pose_detection_confidence=0.3,         # un poco más laxo
            # min_tracking_confidence (opcional; tu módulo lo soporta)
            min_tracking_confidence=0.2,
        )
        pose_landmarker_video = PoseLandmarkerFactory(pose_cfg_video).create_with_fallback()
        logger.info("PoseLandmarker (VIDEO) inicializado.")
    else:
        pose_landmarker_video = None
        logger.info("POSE_USE_VIDEO=0 → WebRTC usará PoseLandmarker (IMAGE).")

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
    logger.info("PoseLandmarkers liberados.")

    try:
        if face_landmarker and hasattr(face_landmarker, "close"):
            face_landmarker.close()
    except Exception:
        pass
    face_landmarker = None
    logger.info("FaceLandmarker liberado.")

    for pc in list(pcs):
        try:
            await pc.close()
        except Exception:
            pass
    pcs.clear()
    results_dc_by_pc.clear()
    ctrl_dc_by_pc.clear()
    need_keyframe_by_pc.clear()
    results_seq_by_pc.clear()
    logger.info("RTCPeerConnections cerrados.")

# ─────────────── Helpers ───────────────
def _decode_image_from_request(request) -> np.ndarray:
    if request.files and "image" in request.files:
        body = request.files["image"][0].body
    elif request.json and "image_b64" in request.json:
        try:
            body = base64.b64decode(request.json["image_b64"])
        except Exception:
            raise InvalidUsage("image_b64 no es base64 válido")
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

# ---- Face JSON serializer ----
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

# ─────────────── Binary packers ───────────────
def pack_pose_frame(image_w: int, image_h: int, poses: List[List[Tuple[int, int]]]) -> bytes:
    out = bytearray()
    out += b"PO"
    out += bytes([0])            # version
    out += bytes([len(poses)])
    out += struct.pack("<HH", image_w, image_h)
    for pts in poses:
        out += bytes([len(pts)])
        for (x, y) in pts:
            out += struct.pack("<HH", max(0, min(65535, x)), max(0, min(65535, y)))
    return bytes(out)

def pack_pose_frame_delta(
    prev: List[List[Tuple[int, int]]] | None,
    curr: List[List[Tuple[int, int]]],
    image_w: int,
    image_h: int,
    keyframe: bool,
    *,
    seq: Optional[int] = None,
    ver: int = 1,  # v1 carries u16 seq after flags
) -> bytes:
    # If absolute body is needed, force the header flag too.
    absolute_needed = (prev is None) or (len(prev) != len(curr))
    keyframe = keyframe or absolute_needed

    out = bytearray(b"PD")
    out += bytes([ver & 0xFF])               # ver
    out += bytes([1 if keyframe else 0])     # flags
    if ver >= 1:
        out += struct.pack("<H", (seq or 0) & 0xFFFF)  # u16 seq

    out += bytes([len(curr)])
    out += struct.pack("<HH", image_w, image_h)

    if keyframe:
        for pts in curr:
            out += bytes([len(pts)])
            for (x, y) in pts:
                out += struct.pack("<HH", x, y)
        return bytes(out)

    for p, cpose in enumerate(curr):
        out += bytes([len(cpose)])
        pmask = 0
        for i, (x, y) in enumerate(cpose):
            px, py = prev[p][i]
            if x != px or y != py:
                pmask |= (1 << i)
        out += struct.pack("<Q", pmask)
        for i, (x, y) in enumerate(cpose):
            if (pmask >> i) & 1:
                dx = max(-127, min(127, x - prev[p][i][0]))
                dy = max(-127, min(127, y - prev[p][i][1]))
                out += struct.pack("<bb", dx, dy)
    return bytes(out)

def _poses_px_from_result(result, img_shape) -> Tuple[int, int, List[List[Tuple[int, int]]]]:
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

# ─────────────── WebSockets (echo / pose / face) ───────────────
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

# ─────────────── WebRTC consume (no remote video back) ───────────────
class PoseTransformTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self, track: MediaStreamTrack, pc: RTCPeerConnection):
        super().__init__()
        self._track = relay.subscribe(track)
        self._pc = pc
    async def recv(self) -> av.VideoFrame:
        frame: av.VideoFrame = await self._track.recv()
        return frame

def _wire_results_dc_handlers(pc, dc):
    @dc.on("open")
    def _on_open():
        logger.info("DataChannel 'results' open on PC %s", id(pc))
    @dc.on("close")
    def _on_close():
        logger.info("DataChannel 'results' closed on PC %s", id(pc))
    @dc.on("message")
    def _on_message(msg):
        # Back-compat: accept "KF" here too, though normal path is 'ctrl'
        try:
            if isinstance(msg, str) and msg.strip().upper() == "KF":
                need_keyframe_by_pc[pc] = True
                logger.info("Client requested keyframe via 'results' (PC %s)", id(pc))
        except Exception:
            pass
    return dc

def _wire_ctrl_dc_handlers(pc, dc):
    @dc.on("open")
    def _on_open():
        logger.info("DataChannel 'ctrl' open on PC %s", id(pc))
    @dc.on("close")
    def _on_close():
        logger.info("DataChannel 'ctrl' closed on PC %s", id(pc))
    @dc.on("message")
    def _on_message(msg):
        try:
            if isinstance(msg, str) and msg.strip().upper() == "KF":
                need_keyframe_by_pc[pc] = True
                logger.info("KF requested via 'ctrl' (PC %s)", id(pc))
        except Exception:
            pass
    return dc

def _make_results_dc(pc: RTCPeerConnection):
    # Unordered + lossy to avoid head-of-line blocking on deltas
    dc = pc.createDataChannel("results", ordered=False, maxRetransmits=0)
    _wire_results_dc_handlers(pc, dc)
    return dc

def _make_ctrl_dc(pc: RTCPeerConnection):
    # Reliable channel for control messages
    dc = pc.createDataChannel("ctrl", ordered=True)
    _wire_ctrl_dc_handlers(pc, dc)
    return dc

def _recycle_results_dc(pc: RTCPeerConnection):
    old = results_dc_by_pc.get(pc)
    try:
        if old:
            old.close()
    except Exception:
        pass
    logger.info("Recreating 'results' DataChannel on PC %s", id(pc))
    dc = _make_results_dc(pc)
    return dc

async def _consume_incoming_video(track: MediaStreamTrack, pc: RTCPeerConnection):
    """Consumes client video, runs pose, and pushes results via DataChannel."""
    global pose_lock, pose_landmarker_video, pose_landmarker_image

    subscribed = relay.subscribe(track)

    # Delta & pacing state
    last_poses_px: List[List[Tuple[int,int]]] | None = None
    last_key_ms = 0
    last_sent_ms = 0
    last_change_ms = 0
    last_abs_ms = 0
    idle_start_ms: Optional[int] = None

    # More aggressive refresh to heal loss/reorder and long stillness.
    KEYFRAME_INTERVAL_MS = 300
    NOCHANGE_KEYFRAME_AFTER_MS = 400
    MIN_SEND_MS = 66  # ~15 fps when healthy

    # ── Congestion control ───────────────────────────────────────────────────
    SEND_THRESHOLD = 32_768
    RECYCLE_AFTER_MS = 300
    congested_since_ms: Optional[int] = None
    # ─────────────────────────────────────────────────────────────────────────

    last_ts_input = 0

    while True:
        try:
            frame: av.VideoFrame = await subscribed.recv()
        except asyncio.CancelledError:
            break
        except Exception:
            break

        dc = results_dc_by_pc.get(pc)
        if not dc or getattr(dc, "readyState", "") != "open":
            continue

        ts_ms = int(time.monotonic() * 1000)

        # Input cadence diagnostics
        if last_ts_input and (ts_ms - last_ts_input) > FRAME_GAP_WARN_MS:
            logger.warning("Input frame gap: %d ms", ts_ms - last_ts_input)

        if ts_ms <= last_ts_input:
            ts_ms = last_ts_input + 1
        last_ts_input = ts_ms

        buf = getattr(dc, "bufferedAmount", 0)

        # Congestion handling & recycle
        if buf >= SEND_THRESHOLD:
            congested_since_ms = congested_since_ms or ts_ms
            if (ts_ms - congested_since_ms) >= RECYCLE_AFTER_MS:
                dc = _recycle_results_dc(pc)
                results_dc_by_pc[pc] = dc
                congested_since_ms = None
                # Force a fresh keyframe after recycle
                last_key_ms = 0
                last_poses_px = None
                need_keyframe_by_pc[pc] = True
            continue
        else:
            congested_since_ms = None

        # Adaptive pacing
        if (ts_ms - last_sent_ms) < MIN_SEND_MS:
            continue

        # latest-only: skip if inference is already running
        if pose_lock is not None and pose_lock.locked():
            continue

        try:
            # === Inference ===
            rgb = frame.to_ndarray(format="rgb24")
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # FIX: derive image shape from the current frame (no img_bgr here)
            h, w = frame.height, frame.width
            img_shape = (h, w, 3)

            async with pose_lock:
                if pose_landmarker_video is not None:
                    result = pose_landmarker_video.detect_for_video(mp_image, ts_ms)
                else:
                    result = pose_landmarker_image.detect(mp_image)

            # === Serialize & send (send once, only if room) ===
            if WEBRTC_JSON_RESULTS:
                payload = _results_pose_to_json(result, img_shape)  # FIX: use img_shape
                payload["ts_ms"] = ts_ms
                if getattr(dc, "bufferedAmount", 0) < SEND_THRESHOLD:
                    dc.send(json.dumps(payload))
                    last_sent_ms = ts_ms
            else:
                w0, h0, poses_px = _poses_px_from_result(result, img_shape)  # FIX: use img_shape

                changed = (poses_px != last_poses_px)
                if changed or last_change_ms == 0:
                    last_change_ms = ts_ms

                # detect idle (no sends or no changes for a while)
                if (ts_ms - last_sent_ms) > IDLE_TO_FORCE_KF_MS or \
                   (ts_ms - last_change_ms) > IDLE_TO_FORCE_KF_MS:
                    idle_start_ms = idle_start_ms or ts_ms
                else:
                    idle_start_ms = None

                # Decide if a keyframe is needed
                external_kf = bool(need_keyframe_by_pc.pop(pc, False))
                force_key   = (last_poses_px is None) or (len(last_poses_px) != len(poses_px))
                gap_key     = (ts_ms - last_sent_ms) > 250
                stale_key   = (ts_ms - last_key_ms) >= KEYFRAME_INTERVAL_MS
                nochange_kf = (ts_ms - last_change_ms) >= NOCHANGE_KEYFRAME_AFTER_MS
                first_move_after_idle = changed and (idle_start_ms is not None)
                heartbeat_abs = ABSOLUTE_INTERVAL_MS > 0 and (ts_ms - last_abs_ms) >= ABSOLUTE_INTERVAL_MS

                need_key = external_kf or force_key or gap_key or stale_key or nochange_kf or first_move_after_idle or heartbeat_abs

                # PD v1 with sequence (lets client detect loss/reorder)
                seq = (results_seq_by_pc.get(pc, 0) + 1) & 0xFFFF
                results_seq_by_pc[pc] = seq

                if "pack_pose_frame_delta" in globals():
                    packet = globals()["pack_pose_frame_delta"](
                        last_poses_px, poses_px, w0, h0, need_key, seq=seq, ver=1
                    )
                    if need_key:
                        last_key_ms = ts_ms
                        last_abs_ms = ts_ms
                else:
                    packet = pack_pose_frame(w0, h0, poses_px)
                    last_key_ms = ts_ms  # 'PO' is absolute
                    last_abs_ms = ts_ms

                if getattr(dc, "bufferedAmount", 0) < SEND_THRESHOLD:
                    dc.send(packet)
                    last_sent_ms = ts_ms
                    last_poses_px = poses_px

        except Exception as e:
            logger.warning("Pose processing error: %s", e)
            continue

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
    """Accepts SDP offer, returns SDP answer. Client adds a video track and data channels."""
    params = request.json or {}
    if "sdp" not in params or "type" not in params:
        raise InvalidUsage("Body JSON must contain 'sdp' and 'type'.")

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection(configuration=_rtc_configuration())
    pcs.add(pc)
    logger.info("Created PC %s", id(pc))

    # Proactively create both DataChannels
    try:
        results_dc_by_pc[pc] = _make_results_dc(pc)  # unordered, lossy
        ctrl_dc_by_pc[pc]    = _make_ctrl_dc(pc)     # reliable
    except Exception:
        pass

    # Init PD sequence for this PC
    results_seq_by_pc[pc] = 0

    @pc.on("datachannel")
    def on_dc(channel):
        label = getattr(channel, "label", "")
        if label == "results":
            results_dc_by_pc[pc] = _wire_results_dc_handlers(pc, channel)
            logger.info("Data channel 'results' open on PC %s", id(pc))
        elif label == "ctrl":
            ctrl_dc_by_pc[pc] = _wire_ctrl_dc_handlers(pc, channel)
            logger.info("Data channel 'ctrl' open on PC %s", id(pc))
        else:
            logger.info("Unknown data channel '%s' on PC %s", label, id(pc))

    @pc.on("track")
    def on_track(track):
        logger.info("Track %s received (kind=%s) on PC %s", track.id, track.kind, id(pc))
        if track.kind == "video":
            asyncio.create_task(_consume_incoming_video(track, pc))

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
                ctrl_dc_by_pc.pop(pc, None)
                need_keyframe_by_pc.pop(pc, None)
                results_seq_by_pc.pop(pc, None)

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
        "Servidor Sanic OK. Prueba /ws, /http, /ws/pose, /ws/face o /webrtc/offer (POST signaling)."
    )

if __name__ == "__main__":
    # Nota: dev=True y debug=True para desarrollo; desactívalo en prod.
    app.run(host="0.0.0.0", port=8000, dev=True, debug=True)
