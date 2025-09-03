# connection/webrtc.py — GStreamer (webrtcbin) version (binary-only PO/PD)
# WITH PRINT LOGS + DIAGNOSTICS — negotiated datachannels (skip DCEP)
# Thread-safety fixes for asyncio <-> GLib (uvloop), safe SDP parsing,
# and appsink->asyncio queue handoff.

from __future__ import annotations

import os
import time
import asyncio
import struct
import threading
import contextlib
import traceback
import inspect
from dataclasses import dataclass
from collections import deque
from typing import Callable, Optional, Dict, Set, List, Tuple, Any, Awaitable

import numpy as np
from sanic import Blueprint, response

# ─────────────── GStreamer / GI ───────────────
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstWebRTC, GstSdp, GstApp, GLib, GObject

from .webrtc_session import GSTWebRTCSession, sessions as _sessions
from .config import *

Gst.init(None)

# ─────────────── Hard mute of external loggers when PRINT_LOGS=0 ───────────────
if not PRINT_LOGS:
    import logging, warnings, sys

    # 1) Disable stdlib logging everywhere
    logging.disable(logging.CRITICAL)
    for name in (
        "asyncio",
        "sanic.root",
        "sanic.error",
        "sanic.access",
        "sanic.server",
        "uvicorn",
        "gunicorn",
        "uvloop",
        "aiohttp",
    ):
        try:
            logging.getLogger(name).disabled = True
            logging.getLogger(name).setLevel(logging.CRITICAL)
        except Exception:
            pass

    # 2) Hide warnings
    warnings.filterwarnings("ignore")

    # 3) Ensure asyncio debug/slow-callback logs are OFF
    try:
        os.environ.pop("PYTHONASYNCIODEBUG", None)
    except Exception:
        pass
    try:
        _loop = asyncio.get_event_loop()
        _loop.set_debug(False)
        if hasattr(_loop, "slow_callback_duration"):
            try:
                _loop.slow_callback_duration = float("inf")
            except Exception:
                pass
    except Exception:
        pass

    # 4) GStreamer/GLib quiet
    try:
        Gst.debug_set_active(False)
        Gst.debug_set_threshold_from_string("0", True)
    except Exception:
        pass

    # 5) Optional nuclear option: also redirect stdout/stderr to /dev/null when HARD_MUTE_STDIO=1
    if os.getenv("HARD_MUTE_STDIO", "0") == "1":
        try:
            devnull = open(os.devnull, "w")
            sys.stdout = devnull
            sys.stderr = devnull
        except Exception:
            pass

# ──────────────────────────────────────────────────────────────────────────────
# Adapter API (modular multi-task):
# - For each task:
#   make_mp_image(rgb_np) -> mp.Image
#   detect_image(mp_image) -> result
#   detect_video(mp_image, ts_ms: int) -> result
#   points_from_result(result, img_shape)
#       -> (w:int, h:int, List[List[(x,y)]]]  # v2 (compat)
#       -> (w:int, h:int, List[List[(x,y,z)]]])  # v3 (nuevo con z)
# If you don't pass adapters, we fallback to the legacy single-task hooks.
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TaskAdapter:
    name: str
    make_mp_image: Callable[[np.ndarray], Any]
    detect_image: Callable[[Any], Awaitable[Any] | Any]
    detect_video: Callable[[Any, int], Awaitable[Any] | Any]
    # Nota: devoluciones XY o XYZ; los empaquetadores detectan si hay z.
    points_from_result: Callable[[Any, tuple[int, int, int]], tuple[int, int, List[List[Tuple[float, ...]]]]]
    log_label: str = "keypoints"


# GStreamer MainLoop (GLib) — ejecutar en 2º hilo
_gst_loop_started = False
_gst_loop = None
_gst_loop_thread: Optional[threading.Thread] = None

# below your other imports
try:
    from .codec import CANDIDATE_DECODERS, CANDIDATE_ENCODERS, _find_first_factory
except Exception:
    # Fallbacks if run as a single file or import path differs
    CANDIDATE_ENCODERS = ["vah265enc", "vaapih265enc", "nvh265enc", "x265enc"]
    CANDIDATE_DECODERS = ["vah265dec", "vaapih265dec", "nvh265dec", "avdec_h265"]

    def _find_first_factory(names):
        for n in names:
            if Gst.ElementFactory.find(n):
                return n
        return None


def _ensure_gst_mainloop():
    global _gst_loop_started, _gst_loop, _gst_loop_thread
    if _gst_loop_started:
        return
    _gst_loop_started = True
    _gst_loop = GLib.MainLoop()
    _ginfo("Starting GstMainLoop thread")

    def _run():
        try:
            _gst_loop.run()
        except Exception as e:
            _gwarn(f"GstMainLoop exited with error: {e}")
        finally:
            _ginfo("GstMainLoop finished")

    _gst_loop_thread = threading.Thread(target=_run, name="GstMainLoop", daemon=True)
    _gst_loop_thread.start()


# Helper to check element availability
def _has_factory(name: str) -> bool:
    return Gst.ElementFactory.find(name) is not None


# ───────── Cuantización fija (px * 2^shift) ─────────
# 0→1x (entero), 1→1/2 px, 2→1/4 px … (limitado a 0..3)
POSE_Q_SHIFT = int(os.getenv("POSE_Q_SHIFT", "1"))
POSE_Q_SHIFT = max(0, min(POSE_Q_SHIFT, 3))
POSE_Q_SCALE = 1 << POSE_Q_SHIFT

# ───────── Cuantización de Z (normalizado) ─────────
# Escala independiente para z (normalizado). 2^7=128 suele ir bien para MediaPipe.
POSE_Z_SHIFT = int(os.getenv("POSE_Z_SHIFT", "7"))
POSE_Z_SHIFT = max(0, min(POSE_Z_SHIFT, 10))
POSE_Z_SCALE = 1 << POSE_Z_SHIFT


def _ver_byte_with_scale(base_ver: int = 2) -> int:
    """
    Byte de versión con la escala XY en los 2 bits bajos (shift).
    bits[1:0] = shift (0→1x,1→2x,2→4x)
    bits[7:2] = base_ver (2=XY, 3=XYZ)
    """
    return (base_ver & 0xFC) | (POSE_Q_SHIFT & 0x03)


def _q16_px(v: float | int) -> int:
    """Convierte px reales a entero uint16 en unidades (px * 2^shift)."""
    return max(0, min(65535, int(round(float(v) * POSE_Q_SCALE))))  # clamp


def _qi16_z(v: float | int) -> int:
    """Convierte z (normalizado/metros) a int16 en unidades (z * 2^POSE_Z_SHIFT)."""
    return max(-32768, min(32767, int(round(float(v) * POSE_Z_SCALE))))


def _has_z(poses: List[List[Tuple[float, ...]]]) -> bool:
    """True si al menos un punto trae componente z (len>=3)."""
    for pts in poses:
        for p in pts:
            if len(p) >= 3:
                return True
    return False


def _quantize_xy(poses_xy: List[List[Tuple[float, float]]]) -> List[List[Tuple[int, int]]]:
    return [[(_q16_px(x), _q16_px(y)) for (x, y) in pts] for pts in poses_xy]


def _quantize_xyz(poses_xyz: List[List[Tuple[float, float, float]]]) -> List[List[Tuple[int, int, int]]]:
    qposes: List[List[Tuple[int, int, int]]] = []
    for pts in poses_xyz:
        qpts: List[Tuple[int, int, int]] = []
        for p in pts:
            x, y = p[0], p[1]
            z = p[2] if len(p) >= 3 else 0.0
            qpts.append((_q16_px(x), _q16_px(y), _qi16_z(z)))
        qposes.append(qpts)
    return qposes


# ─────────────── Empaquetadores binarios (PO/PD) ───────────────
def pack_pose_frame(image_w: int, image_h: int, poses: List[List[Tuple[float, ...]]]) -> bytes:
    """
    PO absoluto con W/H y coordenadas cuantizados.
    v2 (XY): por punto <HH>
    v3 (XYZ): por punto <HHh>
    El byte de versión codifica base_ver (bits 7..2) y shift XY (bits 1..0).
    """
    send_xyz = _has_z(poses)
    base_ver = 3 if send_xyz else 2
    ver = _ver_byte_with_scale(base_ver)
    wq = _q16_px(image_w)
    hq = _q16_px(image_h)

    out = bytearray()
    out += b"PO"
    out += bytes([ver])
    out += struct.pack("<H", min(len(poses), 0xFFFF))
    out += struct.pack("<HH", wq, hq)

    if send_xyz:
        qposes = _quantize_xyz(poses)  # [(xq,yq,zq)]
        for pts in qposes:
            out += struct.pack("<H", min(len(pts), 0xFFFF))
            for (xq, yq, zq) in pts:
                out += struct.pack("<HHh", xq, yq, zq)
    else:
        qposes = _quantize_xy(poses)  # [(xq,yq)]
        for pts in qposes:
            out += struct.pack("<H", min(len(pts), 0xFFFF))
            for (xq, yq) in pts:
                out += struct.pack("<HH", xq, yq)

    return bytes(out)


def pack_pose_frame_delta(
    prev: List[List[Tuple[float, ...]]] | None,
    curr: List[List[Tuple[float, ...]]],
    image_w: int,
    image_h: int,
    keyframe: bool,
    *,
    seq: Optional[int] = None,
    ver: int = 2,
) -> bytes:
    """
    PD con cuantización fija:
      - v2 (XY): mask + (dx:int8, dy:int8)
      - v3 (XYZ): mask + (dx:int8, dy:int8, dz:int8)
    Envía KF si:
      - no hay prev compatible, o contar/longitud difiere, o
      - algún delta cuantizado excede int8 (±127).
    Optimización "no change": si no cambió nada y ABSOLUTE_INTERVAL_MS<=0 → b"".
    """
    # Detecta si debemos enviar XYZ (aunque 'ver'==2 se permite, usamos base>=3 si hay z)
    send_xyz = _has_z(curr) or _has_z(prev or [])
    base_ver = 3 if send_xyz else max(2, ver)
    ver_byte = _ver_byte_with_scale(base_ver)

    wq = _q16_px(image_w)
    hq = _q16_px(image_h)

    if send_xyz:
        qcurr = _quantize_xyz(curr)  # [(xq,yq,zq)]
        qprev = _quantize_xyz(prev) if prev is not None else None
    else:
        qcurr = _quantize_xy(curr)   # [(xq,yq)]
        qprev = _quantize_xy(prev) if prev is not None else None

    # ¿Necesita absoluto?
    absolute_needed = (
        qprev is None
        or (len(qprev) != len(qcurr))
        or any(len(qprev[p]) != len(qcurr[p]) for p in range(len(qcurr)))
    )
    keyframe = keyframe or absolute_needed

    # "No change" (comparando cuantizados) → permite saltarse envío
    if not keyframe and qprev is not None and len(qprev) == len(qcurr):
        any_change = False
        for p in range(len(qcurr)):
            if qcurr[p] != qprev[p]:
                any_change = True
                break
        if (not any_change) and ABSOLUTE_INTERVAL_MS <= 0:
            return b""

    # Cabecera PD
    out = bytearray(b"PD")
    out += bytes([ver_byte])                      # versión (base_ver+shift)
    out += bytes([1 if keyframe else 0])          # flag KF
    if base_ver >= 1:
        out += struct.pack("<H", (seq or 0) & 0xFFFF)
    out += struct.pack("<H", min(len(qcurr), 0xFFFF))
    out += struct.pack("<HH", wq, hq)

    # KF: escribir absolutos cuantizados
    if keyframe:
        for pts in qcurr:
            out += struct.pack("<H", min(len(pts), 0xFFFF))
            if send_xyz:
                for (xq, yq, zq) in pts:
                    out += struct.pack("<HHh", xq, yq, zq)
            else:
                for (xq, yq) in pts:
                    out += struct.pack("<HH", xq, yq)
        return bytes(out)

    # Delta: máscara + deltas int8 (en unidades cuantizadas)
    large_move_detected = False
    deltas_per_pose: List[List[Tuple[int, ...]]] = []

    for p, cpose in enumerate(qcurr):
        pose_deltas: List[Tuple[int, ...]] = []
        pprev = qprev[p]
        for i, cur in enumerate(cpose):
            changed = False
            if send_xyz:
                cx, cy, cz = cur
                px, py, pz = pprev[i]
                dx, dy, dz = cx - px, cy - py, cz - pz
                if dx != 0 or dy != 0 or dz != 0:
                    changed = True
                    if any(v < -127 or v > 127 for v in (dx, dy, dz)):
                        large_move_detected = True
                pose_deltas.append((dx, dy, dz))
            else:
                cx, cy = cur
                px, py = pprev[i]
                dx, dy = cx - px, cy - py
                if dx != 0 or dy != 0:
                    changed = True
                    if dx < -127 or dx > 127 or dy < -127 or dy > 127:
                        large_move_detected = True
                pose_deltas.append((dx, dy))
        deltas_per_pose.append(pose_deltas)

    if large_move_detected:
        # Reintentar como KF absoluto (con la misma versión/escala)
        return pack_pose_frame_delta(prev, curr, image_w, image_h, True, seq=seq, ver=base_ver)

    # Escribir máscaras y deltas (clamp por seguridad)
    for p, cpose in enumerate(qcurr):
        npts = len(cpose)
        out += struct.pack("<H", min(npts, 0xFFFF))
        # Construir máscara de puntos cambiados
        pmask = 0
        if send_xyz:
            for i, (cx, cy, cz) in enumerate(cpose):
                px, py, pz = qprev[p][i]
                if (cx != px) or (cy != py) or (cz != pz):
                    pmask |= (1 << i)
        else:
            for i, (cx, cy) in enumerate(cpose):
                px, py = qprev[p][i]
                if (cx != px) or (cy != py):
                    pmask |= (1 << i)

        mask_bytes = (npts + 7) // 8
        out += int(pmask).to_bytes(mask_bytes, "little", signed=False)

        # Deltas compactos
        for i in range(npts):
            if (pmask >> i) & 1:
                if send_xyz:
                    dx, dy, dz = deltas_per_pose[p][i]
                    dx = max(-127, min(127, dx))
                    dy = max(-127, min(127, dy))
                    dz = max(-127, min(127, dz))
                    out += struct.pack("<bbb", dx, dy, dz)
                else:
                    dx, dy = deltas_per_pose[p][i]
                    dx = max(-127, min(127, dx))
                    dy = max(-127, min(127, dy))
                    out += struct.pack("<bb", dx, dy)

    return bytes(out)


# ─────────────── PyAV-based AV1 decoder check (opcional) ───────────────
def _pyav_has_av1_decoder() -> bool:
    try:
        import av  # lazy

        CodecContext = getattr(av, "CodecContext", None)
        if CodecContext is None:
            return False
        CodecContext.create("av1", "r")
        return True
    except Exception:
        return False


def _ensure_av1_decoder(sample_path: Optional[str] = None, max_frames: int = 3) -> Dict[str, object]:
    info: Dict[str, object] = {}
    try:
        import av

        info["pyav_version"] = getattr(av, "__version__", "?")
    except Exception as e:
        return {"error": f"PyAV not available: {e}"}
    try:
        av.CodecContext.create("av1", "r")
        info["decoder_check"] = "AV1 decoder present"
    except Exception as e:
        info["decoder_check"] = f"AV1 decoder NOT present: {e}"
        return info
    if sample_path and os.path.exists(sample_path):
        try:
            with av.open(sample_path) as cont:
                vstream = next(s for s in cont.streams if s.type == "video")
                info["stream_codec"] = getattr(vstream.codec_context, "name", "?")
                decoded = 0
                for _ in cont.decode(vstream):
                    decoded += 1
                    if decoded >= max_frames:
                        break
                info["decoded_frames"] = decoded
        except Exception as e:
            info["sample_decode_error"] = str(e)
    return info


# ─────────────── Utilidades STUN/TURN para webrtcbin ───────────────
def _fmt_stun(url: str) -> str:
    url = url.strip()
    if url.startswith("stun://"):
        return url
    if url.startswith("stun:"):
        return "stun://" + url[5:]
    return url


def _fmt_turn(turn_url: Optional[str], user: Optional[str], pwd: Optional[str]) -> Optional[str]:
    if not turn_url:
        return None
    scheme = "turn"
    hostpart = turn_url
    if "turns:" in turn_url:
        scheme = "turns"
        hostpart = turn_url.split("turns:", 1)[1]
    elif "turn:" in turn_url:
        hostpart = turn_url.split("turn:", 1)[1]
    auth = ""
    if user and pwd:
        auth = f"{user}:{pwd}@"
    return f"{scheme}://{auth}{hostpart}?transport=udp"

# ─────────────── Constructor del Blueprint WebRTC (Sanic) ───────────────
def build_webrtc_blueprint(
    *,
    # Legacy single-task hooks (back-compat)
    make_mp_image: Callable | None = None,
    detect_image: Callable | None = None,
    detect_video: Callable | None = None,
    poses_px_from_result: Callable | None = None,  # legado (XY)
    # New: multiple adapters
    adapters: Dict[str, TaskAdapter] | None = None,
    default_task: str = "pose",
    url_prefix: str = "",
) -> Blueprint:
    _ensure_gst_mainloop()

    bp = Blueprint("webrtc", url_prefix=url_prefix)

    # Ensure the running loop is also non-debug in Sanic context when PRINT_LOGS=0
    if not PRINT_LOGS:
        try:
            loop = asyncio.get_event_loop()
            loop.set_debug(False)
            if hasattr(loop, "slow_callback_duration"):
                try:
                    loop.slow_callback_duration = float("inf")
                except Exception:
                    pass
        except Exception:
            pass

    @bp.listener("before_server_start")
    async def _mute_sanic_logs(app, loop):
        if not PRINT_LOGS:
            import logging

            for name in ("sanic.root", "sanic.error", "sanic.access", "sanic.server"):
                try:
                    logging.getLogger(name).disabled = True
                    logging.getLogger(name).setLevel(logging.CRITICAL)
                except Exception:
                    pass

    @bp.get("/webrtc/av1/selftest")
    async def av1_selftest(request):
        file_arg = request.args.get("file")
        info = _ensure_av1_decoder(file_arg or AV1_SELFTEST_FILE)
        _ginfo(f"AV1 self-test: {info}")
        return response.json(info)

    @bp.post("/webrtc/offer")
    async def webrtc_offer(request):
        params = request.json or {}
        if "sdp" not in params or "type" not in params:
            _gwarn("Bad /webrtc/offer: missing 'sdp' or 'type'")
            return response.json({"error": "Body JSON must contain 'sdp' and 'type'."}, status=400)
        if params["type"].lower() != "offer":
            _gwarn("Bad /webrtc/offer: type != offer")
            return response.json({"error": "Type must be 'offer'."}, status=400)

        sdp_head = (params.get("sdp") or "")[:512]
        loop = asyncio.get_event_loop()

        # Select adapters: multi-task or legacy single-task fallback
        selected_adapters: List[TaskAdapter]
        if adapters:
            tasks = params.get("tasks")
            if isinstance(tasks, list) and tasks:
                selected_adapters = []
                for t in tasks:
                    key = str(t).lower()
                    if key in adapters:
                        selected_adapters.append(adapters[key])
                if not selected_adapters:
                    return response.json({"error": "no valid tasks in 'tasks'", "allowed": list(adapters)}, status=400)
            else:
                key = str(params.get("task") or default_task).lower()
                if key not in adapters:
                    return response.json({"error": f"unknown task '{key}'", "allowed": list(adapters)}, status=400)
                selected_adapters = [adapters[key]]
        else:
            # Build a single default adapter from legacy hooks
            if not (make_mp_image and detect_image and detect_video and poses_px_from_result):
                return response.json({"error": "No adapters provided and legacy hooks incomplete"}, status=500)
            selected_adapters = [TaskAdapter(
                name="default",
                make_mp_image=make_mp_image,
                detect_image=detect_image,
                detect_video=detect_video,
                points_from_result=poses_px_from_result,  # XY legado
                log_label="default"
            )]

        sess = GSTWebRTCSession(adapters=selected_adapters, loop=loop)
        _sessions.add(sess)
        sess.start()

        try:
            sdp_answer = await sess.accept_offer_and_create_answer(params["sdp"])
        except Exception as e:
            exc_text = _exc_str(e)
            try:
                snap = sess.snapshot()
            except Exception as _e2:
                snap = {"snapshot_error": str(_e2)}

            _gwarn(f"[WebRTC {sess.sid}] Failed to create answer: {e!r}")
            _gwarn(f"[WebRTC {sess.sid}] Traceback:\n{exc_text}")
            _gwarn(f"[WebRTC {sess.sid}] Snapshot: {snap}")

            try:
                await sess.stop()
            except Exception:
                pass
            _sessions.discard(sess)

            return response.json({
                "error": "Failed to create answer",
                "exception": repr(e),
                "traceback": exc_text,
                "sid": getattr(sess, "sid", None),
                "snapshot": snap,
                "sdp_head": sdp_head,
            }, status=500)

        _ginfo(f"[WebRTC {sess.sid}] Answer created and returned")
        return response.json({"sdp": sdp_answer, "type": "answer", "sid": sess.sid})

    @bp.listener("after_server_stop")
    async def _cleanup(app, loop_):
        _ginfo("Server stopping; cleaning sessions")
        for sess in list(_sessions):
            try:
                await sess.stop()
            except Exception:
                pass
            _sessions.discard(sess)
        _ginfo("All sessions cleaned up")

    return bp
