# webrtc_module.py
from __future__ import annotations

import os
import time
import asyncio
import json
import struct
from typing import Callable, Optional, Dict, Set, List, Tuple

from sanic import Blueprint, response

# aiortc / PyAV
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceServer,
    RTCConfiguration,
    MediaStreamTrack,
    RTCRtpReceiver,  # still used to *apply* preferences if available
)
from aiortc.contrib.media import MediaRelay
import av  # frames

# ──────────────────────────────────────────────────────────────────────────────
# Tipos de funciones que inyectará app.py (para desacoplar de MediaPipe)
# - detect_image(mp_image) -> result
# - detect_video(mp_image, ts_ms: int) -> result
# - results_to_json(result, img_shape) -> dict
# - poses_px_from_result(result, img_shape) -> (w, h, List[List[Tuple[int,int]]])
# - make_mp_image(rgb_np) -> mp.Image
# Todas las funciones son síncronas o asíncronas según tu implementación.
# ──────────────────────────────────────────────────────────────────────────────

# ─────────────── Config por ENV ───────────────
WEBRTC_JSON_RESULTS  = os.getenv("WEBRTC_JSON_RESULTS", "0") == "1"  # 0 → binario PD/PO
POSE_USE_VIDEO       = os.getenv("POSE_USE_VIDEO", "0") == "1"
ABSOLUTE_INTERVAL_MS = int(os.getenv("ABSOLUTE_INTERVAL_MS", "0"))
IDLE_TO_FORCE_KF_MS  = int(os.getenv("IDLE_TO_FORCE_KF_MS", "500"))
FRAME_GAP_WARN_MS    = int(os.getenv("FRAME_GAP_WARN_MS", "180"))

STUN_URL  = os.getenv("STUN_URL", "stun:stun.l.google.com:19302")
TURN_URL  = os.getenv("TURN_URL")
TURN_USER = os.getenv("TURN_USERNAME")
TURN_PASS = os.getenv("TURN_PASSWORD")

# Optional AV1 self-test sample file (set via env to decode a few frames at startup)
AV1_SELFTEST_FILE = os.getenv("AV1_SELFTEST_FILE")  # e.g., "samples/av1_sample.mkv"

# ─────────────── Estado WebRTC (aislado aquí) ───────────────
relay = MediaRelay()
pcs: Set[RTCPeerConnection] = set()
results_dc_by_pc: Dict[RTCPeerConnection, object] = {}
ctrl_dc_by_pc: Dict[RTCPeerConnection, object] = {}
need_keyframe_by_pc: Dict[RTCPeerConnection, bool] = {}
results_seq_by_pc: Dict[RTCPeerConnection, int] = {}

# ─────────────── Empaquetadores binarios (PO/PD) ───────────────
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
    ver: int = 2,  # v2: máscara de longitud variable + u16 seq
) -> bytes:
    absolute_needed = (prev is None) or (len(prev) != len(curr))
    keyframe = keyframe or absolute_needed

    out = bytearray(b"PD")
    out += bytes([ver & 0xFF])               # ver
    out += bytes([1 if keyframe else 0])     # flags
    if ver >= 1:
        out += struct.pack("<H", (seq or 0) & 0xFFFF)

    out += bytes([len(curr)])
    out += struct.pack("<HH", image_w, image_h)

    if keyframe:
        for pts in curr:
            out += bytes([len(pts)])
            for (x, y) in pts:
                out += struct.pack("<HH", x, y)
        return bytes(out)

    for p, cpose in enumerate(curr):
        npts = len(cpose)
        out += bytes([npts])
        pmask = 0
        for i, (x, y) in enumerate(cpose):
            px, py = prev[p][i]
            if x != px or y != py:
                pmask |= (1 << i)

        mask_bytes = (npts + 7) // 8
        out += int(pmask).to_bytes(mask_bytes, "little", signed=False)

        for i, (x, y) in enumerate(cpose):
            if (pmask >> i) & 1:
                dx = max(-127, min(127, x - prev[p][i][0]))
                dy = max(-127, min(127, y - prev[p][i][1]))
                out += struct.pack("<bb", dx, dy)
    return bytes(out)

# ─────────────── Utilidad: Config RTC ───────────────
def _rtc_configuration() -> RTCConfiguration:
    ice_servers = [RTCIceServer(urls=STUN_URL)]
    if TURN_URL:
        ice_servers.append(RTCIceServer(urls=TURN_URL, username=TURN_USER, credential=TURN_PASS))
    return RTCConfiguration(iceServers=ice_servers)

# ─────────────── PyAV-based AV1 decoder check ───────────────
def _pyav_has_av1_decoder() -> bool:
    """
    Returns True iff PyAV (FFmpeg build) has an AV1 decoder.
    This replaces using aiortc's getCapabilities() as the source of truth.
    """
    try:
        CodecContext = getattr(av, "CodecContext", None)
        if CodecContext is None:
            return False
        # Will raise if decoder not available:
        CodecContext.create("av1", "r")
        return True
    except Exception:
        return False

# ─────────────── Handlers internos de DC ───────────────
def _wire_results_dc_handlers(pc, dc):
    @dc.on("open")
    def _on_open():
        pass
    @dc.on("close")
    def _on_close():
        pass
    @dc.on("message")
    def _on_message(msg):
        try:
            if isinstance(msg, str) and msg.strip().upper() == "KF":
                need_keyframe_by_pc[pc] = True
        except Exception:
            pass
    return dc

def _wire_ctrl_dc_handlers(pc, dc):
    @dc.on("open")
    def _on_open():
        pass
    @dc.on("close")
    def _on_close():
        pass
    @dc.on("message")
    def _on_message(msg):
        try:
            if isinstance(msg, str) and msg.strip().upper() == "KF":
                need_keyframe_by_pc[pc] = True
        except Exception:
            pass
    return dc

def _make_results_dc(pc: RTCPeerConnection):
    dc = pc.createDataChannel("results", ordered=False, maxRetransmits=0)
    return _wire_results_dc_handlers(pc, dc)

def _make_ctrl_dc(pc: RTCPeerConnection):
    dc = pc.createDataChannel("ctrl", ordered=True)
    return _wire_ctrl_dc_handlers(pc, dc)

def _recycle_results_dc(pc: RTCPeerConnection):
    old = results_dc_by_pc.get(pc)
    try:
        if old:
            old.close()
    except Exception:
        pass
    dc = _make_results_dc(pc)
    return dc

# ─────────────── Pista proxy (si en el futuro necesitas transformar video) ───────────────
class PassthroughVideoTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self, track: MediaStreamTrack):
        super().__init__()
        self._track = relay.subscribe(track)
    async def recv(self) -> av.VideoFrame:
        frame: av.VideoFrame = await self._track.recv()
        return frame

# ─────────────── AV1 self-test helper ───────────────
def _ensure_av1_decoder(sample_path: Optional[str] = None, max_frames: int = 3) -> Dict[str, object]:
    """
    Verifies AV1 decoding availability via PyAV/FFmpeg.
    If a sample_path is provided and exists, decodes a few frames to prove it works.
    """
    info: Dict[str, object] = {"pyav_version": getattr(av, "__version__", "?")}

    # Quick decoder existence check
    try:
        CodecContext = getattr(av, "CodecContext", None)
        if CodecContext is not None:
            CodecContext.create("av1", "r")  # raises if decoder is missing
        info["decoder_check"] = "AV1 decoder present"
    except Exception as e:
        info["decoder_check"] = f"AV1 decoder NOT present: {e}"
        return info

    # Optional: decode a few frames from a file
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

# ─────────────── Constructor del Blueprint WebRTC ───────────────
def build_webrtc_blueprint(
    *,
    make_mp_image: Callable,  # def make_mp_image(rgb_np) -> mp.Image
    detect_image: Callable,   # async def detect_image(mp_image) -> result
    detect_video: Callable,   # async def detect_video(mp_image, ts_ms:int) -> result
    results_to_json: Callable,        # def results_to_json(result, img_shape) -> dict
    poses_px_from_result: Callable,   # def poses_px_from_result(result, img_shape) -> (w,h,poses_px)
    url_prefix: str = "",             # opcional: si quieres montar en /webrtc
) -> Blueprint:

    bp = Blueprint("webrtc", url_prefix=url_prefix)

    @bp.get("/webrtc/av1/selftest")
    async def av1_selftest(request):
        """
        On-demand self-test. You can pass ?file=/abs/path/to/av1.mkv to decode a few frames.
        """
        file_arg = request.args.get("file")
        info = _ensure_av1_decoder(file_arg or AV1_SELFTEST_FILE)
        return response.json(info)

    async def _consume_incoming_video(track: MediaStreamTrack, pc: RTCPeerConnection):
        """Consume video entrante, corre pose y empuja resultados por DataChannel."""
        subscribed = track

        last_poses_px: List[List[Tuple[int,int]]] | None = None
        last_key_ms = 0
        last_sent_ms = 0
        last_change_ms = 0
        last_abs_ms = 0
        idle_start_ms: Optional[int] = None

        KEYFRAME_INTERVAL_MS = 300
        NOCHANGE_KEYFRAME_AFTER_MS = 400
        MIN_SEND_MS = 66  # ~15 fps

        SEND_THRESHOLD = 32_768
        RECYCLE_AFTER_MS = 300
        congested_since_ms: Optional[int] = None

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
            if last_ts_input and (ts_ms - last_ts_input) > FRAME_GAP_WARN_MS:
                pass  # gap observed; no logging
            if ts_ms <= last_ts_input:
                ts_ms = last_ts_input + 1
            last_ts_input = ts_ms

            buf = getattr(dc, "bufferedAmount", 0)

            # Congestión & reciclaje del canal
            if buf >= SEND_THRESHOLD:
                congested_since_ms = congested_since_ms or ts_ms
                if (ts_ms - congested_since_ms) >= RECYCLE_AFTER_MS:
                    dc = _recycle_results_dc(pc)
                    results_dc_by_pc[pc] = dc
                    congested_since_ms = None
                    last_key_ms = 0
                    last_poses_px = None
                    need_keyframe_by_pc[pc] = True
                    continue
                continue
            else:
                congested_since_ms = None

            if (ts_ms - last_sent_ms) < MIN_SEND_MS:
                continue

            # === Inference ===
            try:
                rgb = frame.to_ndarray(format="rgb24")
                mp_image = make_mp_image(rgb)
                h, w = frame.height, frame.width
                img_shape = (h, w, 3)

                if POSE_USE_VIDEO:
                    result = await detect_video(mp_image, ts_ms)
                else:
                    result = await detect_image(mp_image)

                # === Serialize & send ===
                if WEBRTC_JSON_RESULTS:
                    payload = results_to_json(result, img_shape)
                    payload["ts_ms"] = ts_ms
                    if getattr(dc, "bufferedAmount", 0) < SEND_THRESHOLD:
                        dc.send(json.dumps(payload))
                        last_sent_ms = ts_ms
                else:
                    w0, h0, poses_px = poses_px_from_result(result, img_shape)

                    changed = (poses_px != last_poses_px)
                    if changed or last_change_ms == 0:
                        last_change_ms = ts_ms

                    if (ts_ms - last_sent_ms) > IDLE_TO_FORCE_KF_MS or \
                       (ts_ms - last_change_ms) > IDLE_TO_FORCE_KF_MS:
                        idle_start_ms = idle_start_ms or ts_ms
                    else:
                        idle_start_ms = None

                    external_kf = bool(need_keyframe_by_pc.pop(pc, False))
                    force_key   = (last_poses_px is None) or (len(last_poses_px) != len(poses_px))
                    gap_key     = (ts_ms - last_sent_ms) > 250
                    stale_key   = (ts_ms - last_key_ms) >= 300
                    nochange_kf = (ts_ms - last_change_ms) >= 400
                    first_move_after_idle = changed and (idle_start_ms is not None)
                    heartbeat_abs = ABSOLUTE_INTERVAL_MS > 0 and (ts_ms - last_abs_ms) >= ABSOLUTE_INTERVAL_MS

                    need_key = (
                        external_kf or force_key or gap_key or stale_key or
                        nochange_kf or first_move_after_idle or heartbeat_abs
                    )

                    seq = (results_seq_by_pc.get(pc, 0) + 1) & 0xFFFF
                    results_seq_by_pc[pc] = seq

                    if "pack_pose_frame_delta" in globals():
                        packet = pack_pose_frame_delta(
                            last_poses_px, poses_px, w0, h0, need_key, seq=seq, ver=2
                        )
                        if need_key:
                            last_key_ms = ts_ms
                            last_abs_ms = ts_ms
                    else:
                        packet = pack_pose_frame(w0, h0, poses_px)
                        last_key_ms = ts_ms
                        last_abs_ms = ts_ms

                    if getattr(dc, "bufferedAmount", 0) < SEND_THRESHOLD:
                        dc.send(packet)
                        last_sent_ms = ts_ms
                        last_poses_px = poses_px

            except Exception:
                continue

    @bp.post("/webrtc/offer")
    async def webrtc_offer(request):
        params = request.json or {}
        if "sdp" not in params or "type" not in params:
            return response.json({"error": "Body JSON must contain 'sdp' and 'type'."}, status=400)

        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = RTCPeerConnection(configuration=_rtc_configuration())
        pcs.add(pc)

        # Crear DCs de forma proactiva
        try:
            results_dc_by_pc[pc] = _make_results_dc(pc)
            ctrl_dc_by_pc[pc]    = _make_ctrl_dc(pc)
        except Exception:
            pass

        results_seq_by_pc[pc] = 0

        @pc.on("datachannel")
        def on_dc(channel):
            label = getattr(channel, "label", "")
            if label == "results":
                results_dc_by_pc[pc] = _wire_results_dc_handlers(pc, channel)
            elif label == "ctrl":
                ctrl_dc_by_pc[pc] = _wire_ctrl_dc_handlers(pc, channel)

        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                asyncio.create_task(_consume_incoming_video(track, pc))

        @pc.on("connectionstatechange")
        async def on_state_change():
            state = pc.connectionState
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

        # AV1 capability gate via PyAV/FFmpeg
        if not _pyav_has_av1_decoder():
            return response.json(
                {"error": "AV1 decoder not available in PyAV/FFmpeg build on server."},
                status=500,
            )

        # Ensure we have a recvonly transceiver so we can optionally apply preferences.
        if not any(getattr(t, "kind", None) == "video" for t in pc.getTransceivers()):
            pc.addTransceiver("video", direction="recvonly")

        # Prefer AV1 if aiortc exposes it
        try:
            caps = RTCRtpReceiver.getCapabilities("video")
            av1_codecs = [
                c for c in caps.codecs
                if ((getattr(c, "mimeType", "") or "").upper() == "VIDEO/AV1")
            ]
            if av1_codecs:
                for t in pc.getTransceivers():
                    kind = getattr(t, "kind", None) or getattr(t.receiver, "kind", None)
                    if kind == "video":
                        t.setCodecPreferences(av1_codecs)
        except Exception:
            pass

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return response.json({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    # Limpieza al apagar
    @bp.listener("after_server_stop")
    async def _cleanup(app, loop):
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

    return bp
