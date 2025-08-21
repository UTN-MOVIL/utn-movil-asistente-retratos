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
from collections import deque
from typing import Callable, Optional, Dict, Set, List, Tuple

import numpy as np
from sanic import Blueprint, response

# ─────────────── GStreamer / GI ───────────────
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstWebRTC, GstSdp, GstApp, GLib, GObject

from .robust_bytes import _as_bytes
from .decoding import attach_rtp_video_decode_chain  # ← keeps decode chain separate

Gst.init(None)

# ─────────────── Global logging switch ───────────────
# When True → log everything via print; when False → absolutely no console output.
PRINT_LOGS = os.getenv("PRINT_LOGS", "0") == "1"

def _noop(*_args, **_kwargs):
    return None

def _log_print(*args, **kwargs):
    if PRINT_LOGS:
        print(*args, **kwargs)

# ─────────────── Hard mute of external loggers when PRINT_LOGS=0 ───────────────
if not PRINT_LOGS:
    import logging, warnings, sys
    # 1) Disable stdlib logging everywhere
    logging.disable(logging.CRITICAL)
    for name in (
        "asyncio", "sanic.root", "sanic.error", "sanic.access", "sanic.server",
        "uvicorn", "gunicorn", "uvloop", "aiohttp",
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

# ─────────────── Small logging helpers (print-only) ───────────────
def _ts():
    return time.strftime("%H:%M:%S")

def _ginfo(msg: str):
    _log_print(f"Srv 0 {_ts()} INFO:   {msg}", flush=True)

def _gwarn(msg: str):
    _log_print(f"Srv 0 {_ts()} WARN:   {msg}", flush=True)

def _gdebug(msg: str):
    _log_print(f"Srv 0 {_ts()} DEBUG:  {msg}", flush=True)

def _exc_str(e: BaseException) -> str:
    """repr + traceback for returning in JSON and printing."""
    tb = traceback.format_exc()
    return f"{e!r}\n{tb}"

# ──────────────────────────────────────────────────────────────────────────────
# Tipos de funciones inyectadas:
# - detect_image(mp_image) -> result
# - detect_video(mp_image, ts_ms: int) -> result
# - poses_px_from_result(result, img_shape) -> (w, h, List[List[Tuple[int,int]]])
# - make_mp_image(rgb_np) -> mp.Image
# ──────────────────────────────────────────────────────────────────────────────

# ─────────────── Config por ENV ───────────────
POSE_USE_VIDEO        = os.getenv("POSE_USE_VIDEO", "0") == "1"
ABSOLUTE_INTERVAL_MS  = int(os.getenv("ABSOLUTE_INTERVAL_MS", "0"))
IDLE_TO_FORCE_KF_MS   = int(os.getenv("IDLE_TO_FORCE_KF_MS", "500"))
FRAME_GAP_WARN_MS     = int(os.getenv("FRAME_GAP_WARN_MS", "180"))

# ACK opcional (confirma entrega real desde el cliente por 'ctrl')
RESULTS_REQUIRE_ACK   = os.getenv("RESULTS_REQUIRE_ACK", "0") == "1"
ACK_WARN_MS           = int(os.getenv("ACK_WARN_MS", "400"))

STUN_URL  = os.getenv("STUN_URL", "stun:stun.l.google.com:19302")
TURN_URL  = os.getenv("TURN_URL")
TURN_USER = os.getenv("TURN_USERNAME")
TURN_PASS = os.getenv("TURN_PASSWORD")

AV1_SELFTEST_FILE = os.getenv("AV1_SELFTEST_FILE")  # optional file to decode at startup

# NEW: force negotiated (pre-created) data channels to skip DCEP
NEGOTIATED_DCS        = os.getenv("NEGOTIATED_DCS", "1") == "1"
DC_RESULTS_ID         = int(os.getenv("DC_RESULTS_ID", "0"))
DC_CTRL_ID            = int(os.getenv("DC_CTRL_ID", "1"))

# NEW: optional ICE wait time (0 = don't wait, return answer immediately)
WAIT_FOR_ICE_MS       = int(os.getenv("WAIT_FOR_ICE_MS", "0"))  # 0 = don't wait

# ─────────────── Estado global ───────────────
_sessions: Set["GSTWebRTCSession"] = set()

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
    ver: int = 2,
) -> bytes:
    absolute_needed = (prev is None) or (len(prev) != len(curr))
    keyframe = keyframe or absolute_needed

    out = bytearray(b"PD")
    out += bytes([ver & 0xFF])
    out += bytes([1 if keyframe else 0])
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

# ─────────────── Sesión por peer (GStreamer WebRTC) ───────────────
class GSTWebRTCSession:
    def __init__(
        self,
        *,
        make_mp_image: Callable,
        detect_image: Callable,
        detect_video: Callable,
        poses_px_from_result: Callable,
        loop: asyncio.AbstractEventLoop,
    ):
        self.loop = loop

        self.make_mp_image = make_mp_image
        self.detect_image = detect_image
        self.detect_video = detect_video
        self.poses_px_from_result = poses_px_from_result

        self.pipeline: Optional[Gst.Pipeline] = None
        self.webrtc: Optional[Gst.Element] = None

        self.results_dc: Optional[GstWebRTC.WebRTCDataChannel] = None
        self.ctrl_dc: Optional[GstWebRTC.WebRTCDataChannel] = None

        # Create asyncio primitives on the right loop/thread
        self.frame_q: asyncio.Queue = asyncio.Queue(maxsize=1)
        self.process_task: Optional[asyncio.Task] = None

        self.last_poses_px: Optional[List[List[Tuple[int, int]]]] = None
        self.last_key_ms: int = 0
        self.last_sent_ms: int = 0
        self.last_change_ms: int = 0
        self.last_abs_ms: int = 0
        self.idle_start_ms: Optional[int] = None
        self.need_keyframe: bool = False
        self.seq: int = 0
        self.last_ts_input: int = 0

        # ACK tracking (seq -> ts_ms)
        self._awaiting_ack: Dict[int, int] = {}
        self._ack_warned: Set[int] = set()
        self.last_ack_seq: Optional[int] = None

        # CREATE the futures on the provided loop (not the GLib thread)
        self._gathering_done = self.loop.create_future()
        self._local_answer_set = self.loop.create_future()  # NEW: to wait for set-local-description

        # Stats
        self.sid = f"{id(self) & 0xFFFFFF:06x}"
        self.stats: Dict[str, int | float] = dict(
            samples_in=0,
            frames_sent=0,
            kf_sent=0,
            delta_sent=0,
            bytes_sent=0,
            drops_due_buffer=0,
            dc_recycles=0,  # maintained for backward compat; no actual recycle on negotiated
            infer_ms_last=0.0,
            infer_ms_avg=0.0,
            acks=0,
            ack_rtt_ms_avg=0.0,
        )
        # ── Appsink / processing visibility
        self._appsink_n = 0
        self._appsink_last_cb_ms = 0
        self._appsink_last_pts_ns: Optional[int] = None
        self._appsink_caps_sig: Optional[str] = None
        self._proc_n = 0

        # ── New: ring buffer for bus diagnostics
        self._bus_tail: deque[str] = deque(maxlen=50)

        self._info("New session created")

    # ─────────────── session-scope print helpers ───────────────
    def _info(self, msg: str):
        if PRINT_LOGS:
            _log_print(f"Srv 0 {_ts()} INFO:   [WebRTC {self.sid}] {msg}", flush=True)
    def _warn(self, msg: str):
        if PRINT_LOGS:
            _log_print(f"Srv 0 {_ts()} WARN:   [WebRTC {self.sid}] {msg}", flush=True)
    def _dbg(self, msg: str):
        if PRINT_LOGS:
            _log_print(f"Srv 0 {_ts()} DEBUG:  [WebRTC {self.sid}] {msg}", flush=True)

    def _buslog(self, level: str, src: str, text: str):
        line = f"{level} [{src}] {text}"
        self._bus_tail.append(line)

    def _mark_gathering_done(self):
        if not self._gathering_done.done():
            self._gathering_done.set_result(True)

    def _mark_local_answer_set(self):  # NEW helper
        if not self._local_answer_set.done():
            self._local_answer_set.set_result(True)

    def _start_processing_task(self):
        if not self.process_task or self.process_task.done():
            self._info("Starting frame processing task")
            self.process_task = self.loop.create_task(self._process_frames())

    def _enqueue_frame(self, frame_np, pts_ns, w, h):
        try:
            self.frame_q.put_nowait((frame_np, pts_ns))
        except asyncio.QueueFull:
            with contextlib.suppress(Exception):
                _ = self.frame_q.get_nowait()   # drop oldest
            self.frame_q.put_nowait((frame_np, pts_ns))
            self._warn("Frame queue full; dropped oldest (keep-latest policy)")

    # ---- pad-buffer probe helper (print-only)
    def _add_buf_probe(self, elem: Gst.Element, label: str, pad_name: str = "src"):
        try:
            pad = elem.get_static_pad(pad_name)
            if not pad:
                self._warn(f"PROBE {label}: no '{pad_name}' pad")
                return
            counter = {"n": 0}
            def _probe(_pad, info):
                if not info or not (info.type & Gst.PadProbeType.BUFFER):
                    return Gst.PadProbeReturn.OK
                buf = info.get_buffer()
                counter["n"] += 1
                if counter["n"] <= 5 or (counter["n"] % 30) == 0:
                    pts_ns = buf.pts if (buf and buf.pts is not None) else -1
                    pts_ms = (pts_ns / 1_000_000.0) if pts_ns >= 0 else -1.0
                    caps = _pad.get_current_caps()
                    caps_s = caps.to_string() if caps else "n/a"
                    self._dbg(f"PROBE {label} #{counter['n']}: pts={pts_ms:.1f}ms caps={caps_s}")
                return Gst.PadProbeReturn.OK
            pad.add_probe(Gst.PadProbeType.BUFFER, _probe)
            self._dbg(f"PROBE installed on {label} ({elem.name}:{pad_name})")
        except Exception as e:
            self._warn(f"PROBE {label} install error: {e}")

    # ---- NEW: sink-pad probe (to see if elements are fed)
    def _add_sink_probe(self, elem: Gst.Element, label: str, pad_name: str = "sink"):
        try:
            pad = elem.get_static_pad(pad_name)
            if not pad: return
            counter = {"n": 0}
            def _probe(_pad, info):
                if not info or not (info.type & Gst.PadProbeType.BUFFER):
                    return Gst.PadProbeReturn.OK
                counter["n"] += 1
                if counter["n"] <= 5 or (counter["n"] % 30) == 0:
                    self._dbg(f"PROBE {label} (sink) #{counter['n']}")
                return Gst.PadProbeReturn.OK
            pad.add_probe(Gst.PadProbeType.BUFFER, _probe)
        except Exception as e:
            self._warn(f"PROBE {label} (sink) install error: {e}")

    # ───── Pipeline / webrtcbin setup ─────
    def _build(self):
        self.pipeline = Gst.Pipeline.new(None)
        self.webrtc = Gst.ElementFactory.make("webrtcbin", "webrtcbin")
        assert self.webrtc is not None, "webrtcbin plugin not available"

        self.webrtc.set_property("latency", 0)

        # STUN/TURN
        self.webrtc.set_property("stun-server", _fmt_stun(STUN_URL))
        turl = _fmt_turn(TURN_URL, TURN_USER, TURN_PASS)
        if turl:
            self.webrtc.set_property("turn-server", turl)
        self._info(f"Using STUN={_fmt_stun(STUN_URL)} TURN={'set' if turl else 'False'}")

        # Bundle policy
        self.webrtc.set_property("bundle-policy", GstWebRTC.WebRTCBundlePolicy.MAX_BUNDLE)

        self.pipeline.add(self.webrtc)

        # Signals
        self.webrtc.connect("on-data-channel", self._on_data_channel)
        self.webrtc.connect("pad-added", self._on_incoming_pad)
        self.webrtc.connect("on-ice-candidate", self._on_ice_candidate)
        self.webrtc.connect("notify::ice-gathering-state", self._on_gathering_state)
        self.webrtc.connect("notify::connection-state", self._on_conn_state)

        # Watch pipeline bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        # IMPORTANT: Do NOT pre-create negotiated DCs here anymore.
        # We'll create them *after* set-remote-description and *before* create-answer.
        if NEGOTIATED_DCS:
            self._info("Negotiated DCs will be created after SRD (skip DCEP)")
        else:
            self._info("NEGOTIATED_DCS=0 → will rely on remote-announced (DCEP) channels")

        self._info("Pipeline constructed")

    # Create negotiated channels (id=0 results, id=1 ctrl). Call only after SRD.
    def _precreate_negotiated_dcs(self):
        if not self.webrtc:
            return
        if self.results_dc or self.ctrl_dc:
            # Already created
            return
        try:
            # results: unordered + lossy, negotiated id
            opts_res = Gst.Structure.new_empty("application/webrtc-data-channel")
            opts_res.set_value("ordered", False)
            opts_res.set_value("max-retransmits", 0)
            opts_res.set_value("negotiated", True)
            opts_res.set_value("id", DC_RESULTS_ID)
            self.results_dc = self.webrtc.emit("create-data-channel", "results", opts_res)
            self._wire_results_dc(self.results_dc)
            self._info(f"Created negotiated DC 'results' id={DC_RESULTS_ID} (unordered, maxRetransmits=0)")
        except Exception as e:
            self._warn(f"Failed to create negotiated 'results' datachannel: {e}")

        try:
            # ctrl: reliable, negotiated id
            opts_ctrl = Gst.Structure.new_empty("application/webrtc-data-channel")
            opts_ctrl.set_value("ordered", True)  # reliable default
            opts_ctrl.set_value("negotiated", True)
            opts_ctrl.set_value("id", DC_CTRL_ID)
            self.ctrl_dc = self.webrtc.emit("create-data-channel", "ctrl", opts_ctrl)
            self._wire_ctrl_dc(self.ctrl_dc)
            self._info(f"Created negotiated DC 'ctrl' id={DC_CTRL_ID} (reliable)")
        except Exception as e:
            self._warn(f"Failed to create negotiated 'ctrl' datachannel: {e}")

    def _on_bus_message(self, bus: Gst.Bus, msg: Gst.Message):
        try:
            t = msg.type
            src = msg.src
            src_name = src.name if isinstance(src, Gst.Element) else "?"
            if t == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                text = f"ERROR: {err} debug={dbg or ''}"
                self._buslog("ERROR", src_name, text)
                self._warn(f"GStreamer ERROR: {err} (debug: {dbg})")
            elif t == Gst.MessageType.WARNING:
                w, dbg = msg.parse_warning()
                text = f"WARNING: {w} debug={dbg or ''}"
                self._buslog("WARNING", src_name, text)
                self._warn(f"GStreamer WARNING: {w} (debug: {dbg})")
            elif t == Gst.MessageType.EOS:
                self._buslog("INFO", src_name, "EOS")
                self._info("Pipeline EOS")
            elif t == Gst.MessageType.STATE_CHANGED:
                if isinstance(src, Gst.Element):
                    old, new, _pending = msg.parse_state_changed()
                    if src is self.pipeline or new in (Gst.State.PAUSED, Gst.State.PLAYING):
                        self._dbg(f"STATE {src.name}: {old.value_nick}->{new.value_nick}")
                        self._buslog("STATE", src.name, f"{old.value_nick}->{new.value_nick}")
        except Exception:
            pass
        return

    def start(self):
        if not self.pipeline:
            self._build()
        self.pipeline.set_state(Gst.State.PLAYING)
        self._info("Pipeline PLAYING")

    async def stop(self):
        self._info("Stopping session")
        try:
            if self.process_task:
                self.process_task.cancel()
                with contextlib.suppress(Exception):
                    await self.process_task
        except Exception as e:
            self._warn(f"Error awaiting process_task cancel: {e}")
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        self.pipeline = None
        self.webrtc = None
        self.results_dc = None
        self.ctrl_dc = None
        self._info("Session stopped")

    # ───── Signaling (HTTP) helpers ─────
    async def accept_offer_and_create_answer(self, offer_sdp_text: str) -> str:
        assert self.webrtc is not None

        self._info("Accepting offer")
        ret, sdpmsg = GstSdp.sdp_message_new()
        if ret != GstSdp.SDPResult.OK:
            raise RuntimeError(f"sdp_message_new failed: {ret}")
        ret = GstSdp.sdp_message_parse_buffer(offer_sdp_text.encode("utf-8"), sdpmsg)
        if ret != GstSdp.SDPResult.OK:
            raise RuntimeError(f"Bad SDP offer: {ret}")

        offer = GstWebRTC.WebRTCSessionDescription.new(
            GstWebRTC.WebRTCSDPType.OFFER, sdpmsg
        )

        # 1) set-remote-description, and only after SRD create negotiated DCs, then create-answer
        def _on_answer_created(promise, webrtc, self_ref: "GSTWebRTCSession"):
            try:
                reply = promise.get_reply()
                answer = reply.get_value("answer")
                p2 = Gst.Promise.new()
                webrtc.emit("set-local-description", answer, p2)
                p2.interrupt()
                self_ref._info("Local SDP answer set")
                try:
                    # mark from GLib thread
                    self_ref.loop.call_soon_threadsafe(self_ref._mark_local_answer_set)
                except Exception:
                    pass
            except Exception as e:
                self_ref._warn(f"create-answer/set-local failed: {e!r}")
                self_ref._buslog("ERROR", "webrtcbin", f"create-answer/set-local failed: {e!r}")

        def _on_srd_applied(promise, webrtc, self_ref: "GSTWebRTCSession"):
            try:
                if NEGOTIATED_DCS:
                    self_ref._precreate_negotiated_dcs()
                else:
                    self_ref._info("NEGOTIATED_DCS=0 → will rely on remote-announced (DCEP) channels")
            except Exception as e:
                self_ref._warn(f"Creating negotiated DCs after SRD failed: {e}")
            # Now generate the answer
            prom2 = Gst.Promise.new_with_change_func(_on_answer_created, webrtc, self_ref)
            webrtc.emit("create-answer", None, prom2)

        prom_srd = Gst.Promise.new_with_change_func(_on_srd_applied, self.webrtc, self)
        self.webrtc.emit("set-remote-description", offer, prom_srd)

        # 2) Wait for local answer to be set (briefly)
        try:
            await asyncio.wait_for(self._local_answer_set, timeout=3.0)
        except asyncio.TimeoutError:
            self._warn("Timeout waiting for local answer; continuing anyway")

        # 3) (Optional) wait a bit for ICE gathering; 0 = skip
        if WAIT_FOR_ICE_MS > 0:
            try:
                await asyncio.wait_for(self._gathering_done, timeout=WAIT_FOR_ICE_MS / 1000.0)
                self._dbg("ICE gathering state: complete")
            except asyncio.TimeoutError:
                self._warn(f"ICE gathering timeout after {WAIT_FOR_ICE_MS}ms; returning early")

        # Get local description
        local: GstWebRTC.WebRTCSessionDescription = self.webrtc.get_property("local-description")
        if not local:
            raise RuntimeError("local-description is None after create-answer; see bus_tail")
        return local.sdp.as_text()

    # ───── DataChannel wiring ─────
    def _wire_results_dc(self, dc: GstWebRTC.WebRTCDataChannel | None):
        if not dc:
            return
        def on_open(ch):
            self._info("DataChannel 'results' open")
            try:
                ch.emit("send-string", "HELLO_FROM_SERVER")
            except Exception as e:
                self._warn(f"send-string hello failed: {e}")
            try:
                pkt = b"PO" + bytes([0, 0]) + struct.pack("<HH", 1, 1)
                gbytes = GLib.Bytes(pkt)
                ch.emit("send-data", gbytes)
                self._dbg("Sent dummy PO packet on 'results' (1x1, 0 poses)")
            except Exception as e:
                self._warn(f"send-data dummy failed: {e}")

        def on_close(ch):
            self._warn("DataChannel 'results' closed")
        def on_error(ch, err):
            self._warn(f"DataChannel 'results' error: {err}")

        def _on_msg_str(ch, msg):
            if isinstance(msg, str) and msg.strip().upper() == "KF":
                self._info("Received KF on 'results' (string) → will keyframe next send")
                self.need_keyframe = True
        def _on_msg_bin(ch, data):
            try:
                b = _as_bytes(data)
                if b and b.strip().upper() == b"KF":
                    self._info("Received KF on 'results' (binary) → will keyframe next send")
                    self.need_keyframe = True
            except Exception:
                pass

        dc.connect("on-open", on_open)
        dc.connect("on-close", on_close)
        dc.connect("on-error", on_error)
        dc.connect("on-message-string", _on_msg_str)
        dc.connect("on-message-data", _on_msg_bin)

    def _handle_ack(self, seq: int):
        ts_now = int(time.monotonic() * 1000)
        if seq in self._awaiting_ack:
            rtt = ts_now - self._awaiting_ack.pop(seq)
            self._ack_warned.discard(seq)
            self.last_ack_seq = seq
            prev = float(self.stats["ack_rtt_ms_avg"])
            self.stats["ack_rtt_ms_avg"] = prev * 0.9 + float(rtt) * 0.1
            self.stats["acks"] = int(self.stats["acks"]) + 1
            self._info(f"ACK received for seq={seq} rtt={rtt}ms avg={self.stats['ack_rtt_ms_avg']:.1f}ms")

    def _wire_ctrl_dc(self, dc: GstWebRTC.WebRTCDataChannel | None):
        if not dc:
            return
        def on_open(ch):
            self._info("DataChannel 'ctrl' open")
        def on_close(ch):
            self._warn("DataChannel 'ctrl' closed")
        def on_error(ch, err):
            self._warn(f"DataChannel 'ctrl' error: {err}")

        def _parse_ack_string(s: str) -> Optional[int]:
            s = s.strip()
            if not s.upper().startswith("ACK"):
                return None
            for sep in (" ", ":"):
                if sep in s:
                    try:
                        return int(s.split(sep, 1)[1].strip())
                    except Exception:
                        return None
            return None

        def _on_msg_str(ch, msg):
            if not isinstance(msg, str):
                return
            if msg.strip().upper() == "KF":
                self._info("Received KF on 'ctrl' (string) → will keyframe next send")
                self.need_keyframe = True
                return
            seq = _parse_ack_string(msg)
            if seq is not None:
                self._dbg(f"ACK (string) received: seq={seq}")
                self._handle_ack(seq)

        def _on_msg_bin(ch, data):
            try:
                b = _as_bytes(data)
                if not b:
                    return
                ub = b.upper()
                if ub == b"KF":
                    self._info("Received KF on 'ctrl' (binary) → will keyframe next send")
                    self.need_keyframe = True
                    return
                if len(b) >= 5 and ub.startswith(b"ACK"):
                    seq = int.from_bytes(b[3:5], "little", signed=False)
                    self._dbg(f"ACK (binary) received: seq={seq}")
                    self._handle_ack(seq)
            except Exception as e:
                self._warn(f"Error parsing 'ctrl' binary msg: {e}")

        dc.connect("on-open", on_open)
        dc.connect("on-close", on_close)
        dc.connect("on-error", on_error)
        dc.connect("on-message-string", _on_msg_str)
        dc.connect("on-message-data", _on_msg_bin)

    def _on_data_channel(self, webrtc, channel: GstWebRTC.WebRTCDataChannel):
        # For DCEP-created channels (NEGOTIATED_DCS=0). Negotiated channels we create locally.
        label = channel.props.label or ""
        self._dbg(f"on-data-channel: '{label}' readyState={channel.get_property('ready-state')}")
        if label == "results":
            self.results_dc = channel
            self._wire_results_dc(channel)
        elif label == "ctrl":
            self.ctrl_dc = channel
            self._wire_ctrl_dc(channel)

    # ───── ICE helpers ─────
    def _on_ice_candidate(self, webrtc, mlineindex, candidate):
        self._dbg(f"ICE candidate mline={mlineindex} cand='{candidate[:64]}...'")

    def _on_gathering_state(self, webrtc, _pspec):
        state = self.webrtc.get_property("ice-gathering-state")
        self._dbg(f"ICE gathering state: {state.value_nick}")
        if state == GstWebRTC.WebRTCICEGatheringState.COMPLETE:
            try:
                self.loop.call_soon_threadsafe(self._mark_gathering_done)
            except Exception:
                pass

    def _on_conn_state(self, webrtc, _pspec):
        state = self.webrtc.get_property("connection-state")
        self._info(f"PC state: {state.value_nick}")
        if state in (
            GstWebRTC.WebRTCPeerConnectionState.FAILED,
            GstWebRTC.WebRTCPeerConnectionState.CLOSED,
            GstWebRTC.WebRTCPeerConnectionState.DISCONNECTED,
        ):
            try:
                _sessions.discard(self)
            except Exception:
                pass
            try:
                if self.pipeline:
                    self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass

    # ───── Incoming media handling ─────
    def _on_incoming_pad(self, webrtc, pad: Gst.Pad):
        # Be defensive about caps.
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps or caps.get_size() == 0:
            self._warn("Incoming pad without caps; ignoring")
            return
        s = caps.get_structure(0)
        if s.get_name() != "application/x-rtp":
            return
        media = (s.get_string("media") or "").lower()
        enc = (s.get_string("encoding-name") or "").upper()
        if media != "video":
            return

        self._info(f"Incoming RTP video pad; encoding={enc} pad={pad.get_name()}")

        # (Optional) quickly count incoming RTP buffers
        try:
            counter = {"n": 0}
            def _rtp_probe(_pad, info):
                if not info or not (info.type & Gst.PadProbeType.BUFFER):
                    return Gst.PadProbeReturn.OK
                counter["n"] += 1
                if counter["n"] <= 5 or (counter["n"] % 50) == 0:
                    self._dbg(f"RTP PAD buf #{counter['n']} on {pad.name} (encoding={enc})")
                return Gst.PadProbeReturn.OK
            pad.add_probe(Gst.PadProbeType.BUFFER, _rtp_probe)
        except Exception as e:
            self._warn(f"Couldn't add RTP pad probe: {e}")

        # Helper to fall back to the original direct attach
        def _fallback_direct_attach():
            self._warn("Falling back to direct attach (no jbuf/queue)")
            try:
                attach_rtp_video_decode_chain(
                    pipeline=self.pipeline,
                    src_pad=pad,
                    encoding_name=enc,
                    on_new_sample=self._on_new_sample,
                    dbg=(self._dbg if PRINT_LOGS else _noop),
                    warn=self._warn,
                )
                self._info("Appsink wired (fallback); waiting for decoded RGB frames…")
                if not self.process_task:
                    self.loop.call_soon_threadsafe(self._start_processing_task)
            except Exception as e:
                self._warn(f"attach decode chain (fallback) failed: {e}")

        # Build ultra-low-latency chain: rtpjitterbuffer → leaky queue → decode
        try:
            jbuf = Gst.ElementFactory.make("rtpjitterbuffer", None)
            q    = Gst.ElementFactory.make("queue", None)
            if not jbuf or not q:
                raise RuntimeError("Failed to create rtpjitterbuffer/queue")

            # Configure for low latency
            jbuf.set_property("latency", 0)
            jbuf.set_property("drop-on-late", True)
            jbuf.set_property("do-lost", True)

            # NEW ↓
            # Do not wait for retransmissions; keep minimal reordering window.
            if jbuf.find_property("do-retransmission"):
                jbuf.set_property("do-retransmission", False)
            if jbuf.find_property("max-reorder"):          # packets
                jbuf.set_property("max-reorder", 8)        # small but safe
            if jbuf.find_property("max-dropout-time"):     # ms
                jbuf.set_property("max-dropout-time", 50)  # don’t wait long on timestamp jumps

            q.set_property("leaky", 2)  # downstream
            q.set_property("max-size-buffers", 1)
            q.set_property("max-size-time", 0)
            q.set_property("max-size-bytes", 0)

            # Add to pipeline and sync states (pipeline is already PLAYING)
            self.pipeline.add(jbuf)
            self.pipeline.add(q)
            if not jbuf.sync_state_with_parent():
                self._warn("jitterbuffer failed to sync state with parent")
            if not q.sync_state_with_parent():
                self._warn("queue failed to sync state with parent")

            # Link: webrtcbin RTP src pad → jbuf → queue
            linkret = pad.link(jbuf.get_static_pad("sink"))
            if linkret != Gst.PadLinkReturn.OK:
                # Clean up and fallback
                self._warn(f"Link webrtcbin:src → rtpjitterbuffer:sink failed: {linkret.value_nick}")
                try:
                    self.pipeline.remove(jbuf)
                    self.pipeline.remove(q)
                except Exception:
                    pass
                _fallback_direct_attach()
                return

            if not jbuf.link(q):
                self._warn("Link rtpjitterbuffer → queue failed")
                try:
                    pad.unlink(jbuf.get_static_pad("sink"))
                except Exception:
                    pass
                try:
                    self.pipeline.remove(jbuf)
                    self.pipeline.remove(q)
                except Exception:
                    pass
                _fallback_direct_attach()
                return

            self._dbg("Inserted rtpjitterbuffer + leaky queue before decode chain (links OK)")

            # Feed the decode chain from the queue's SRC pad
            attach_rtp_video_decode_chain(
                pipeline=self.pipeline,
                src_pad=q.get_static_pad("src"),  # ← feed from the queue now
                encoding_name=enc,
                on_new_sample=self._on_new_sample,
                dbg=(self._dbg if PRINT_LOGS else _noop),
                warn=self._warn,
            )
            self._info("Appsink wired; waiting for decoded RGB frames…")

            if not self.process_task:
                self.loop.call_soon_threadsafe(self._start_processing_task)
        except Exception as e:
            self._warn(f"Failed to attach decode chain via jbuf/queue: {e}")
            _fallback_direct_attach()

    # appsink callback (GStreamer thread) — copy frame & enqueue via loop.call_soon_threadsafe
    def _on_new_sample(self, sink: GstApp.AppSink):
        try:
            now_ms = int(time.monotonic() * 1000)

            sample: Gst.Sample = sink.emit("pull-sample")
            buf: Gst.Buffer = sample.get_buffer()
            caps: Gst.Caps = sample.get_caps()
            s = caps.get_structure(0)
            w = int(s.get_value("width"))
            h = int(s.get_value("height"))
            fmt = s.get_string("format") or "?"

            caps_sig = f"{w}x{h}/{fmt}"
            if caps_sig != self._appsink_caps_sig:
                self._appsink_caps_sig = caps_sig
                self._info(f"APPSINK caps: {caps.to_string()}")

            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                self._warn("Failed to map buffer from appsink")
                return Gst.FlowReturn.ERROR
            try:
                payload = bytes(mapinfo.data)  # one compact copy
                pts_ns = int(buf.pts) if buf.pts is not None and buf.pts >= 0 else -1
            finally:
                buf.unmap(mapinfo)

            dcb_ms = (now_ms - self._appsink_last_cb_ms) if self._appsink_last_cb_ms else 0
            self._appsink_last_cb_ms = now_ms

            pts_ms = (pts_ns / 1_000_000.0) if pts_ns >= 0 else -1.0
            dpts_ms = None
            if self._appsink_last_pts_ns is not None and pts_ns >= 0:
                dpts_ms = (pts_ns - self._appsink_last_pts_ns) / 1_000_000.0
            self._appsink_last_pts_ns = pts_ns

            self._appsink_n += 1

            if FRAME_GAP_WARN_MS > 0 and dcb_ms and dcb_ms > FRAME_GAP_WARN_MS:
                self._warn(f"APPSINK gap Δcb={dcb_ms}ms (> {FRAME_GAP_WARN_MS}ms)")

            if self._appsink_n <= 5 or (self._appsink_n % 30) == 0:
                qsz = self.frame_q.qsize()
                dpts_str = f"{dpts_ms:.1f}ms" if dpts_ms is not None else "n/a"
                self._dbg(
                    f"APPSINK sample #{self._appsink_n}: {w}x{h}/{fmt} "
                    f"pts={pts_ms:.1f}ms Δcb={dcb_ms}ms Δpts={dpts_str} q={qsz}"
                )

            try:
                self.loop.call_soon_threadsafe(self._enqueue_frame, (payload, w, h), pts_ns, w, h)
            except Exception as e:
                self._warn(f"call_soon_threadsafe enqueue failed: {e}")

            return Gst.FlowReturn.OK
        except Exception as e:
            self._warn(f"appsink new-sample error: {e}")
            return Gst.FlowReturn.ERROR

    # main async loop to process frames (binary-only)
    async def _process_frames(self):
        SEND_THRESHOLD = 32_768
        RECYCLE_AFTER_MS = 300
        MIN_SEND_MS = 33  # ~30 fps

        congested_since_ms: Optional[int] = None

        while True:
            try:
                (payload, w, h), pts_ns = await self.frame_q.get()
                # build np view only now (no extra copy)
                frame = np.frombuffer(payload, dtype=np.uint8).reshape((h, w, 3))
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._warn(f"frame_q.get error: {e}")
                continue

            self._proc_n += 1
            if self._proc_n <= 5 or (self._proc_n % 30) == 0:
                dc_state = (self.results_dc.get_property("ready-state")
                            if self.results_dc else None)
                self._dbg(f"PROCESS sample #{self._proc_n}: dequeued; dc_state={dc_state} q={self.frame_q.qsize()}")

            ts_ms = int(time.monotonic() * 1000)
            if ts_ms <= self.last_ts_input:
                ts_ms = self.last_ts_input + 1
            self.last_ts_input = ts_ms

            if RESULTS_REQUIRE_ACK and self._awaiting_ack:
                to_warn = [s for s, t0 in list(self._awaiting_ack.items()) if (ts_ms - t0) >= ACK_WARN_MS and s not in self._ack_warned]
                for s in to_warn:
                    self._ack_warned.add(s)
                    self._warn(f"ACK overdue for seq={s} > {ACK_WARN_MS}ms")

            dc = self.results_dc
            if not dc or dc.get_property("ready-state") != GstWebRTC.WebRTCDataChannelState.OPEN:
                continue

            buf_amt = dc.get_property("buffered-amount") or 0

            if buf_amt >= SEND_THRESHOLD:
                # For negotiated channels, don't try to recreate with the same id.
                congested_since_ms = congested_since_ms or ts_ms
                if (ts_ms - (congested_since_ms or ts_ms)) >= RECYCLE_AFTER_MS:
                    self._info("Results DC congested; will skip frames and force next send as keyframe (no DC recycle for negotiated)")
                    congested_since_ms = None
                    self.last_key_ms = 0
                    self.last_poses_px = None
                    self.need_keyframe = True
                self.stats["drops_due_buffer"] = int(self.stats["drops_due_buffer"]) + 1
                self._dbg(f"Dropping frame due to buffered-amount={buf_amt}")
                continue
            else:
                congested_since_ms = None

            if (ts_ms - self.last_sent_ms) < MIN_SEND_MS:
                continue

            try:
                t0 = time.perf_counter()
                mp_image = self.make_mp_image(frame)
                img_shape = (frame.shape[0], frame.shape[1], 3)

                fn, args = (
                    (self.detect_video, (mp_image, ts_ms))
                    if POSE_USE_VIDEO
                    else (self.detect_image, (mp_image,))
                )

                if inspect.iscoroutinefunction(fn):
                    result = await fn(*args)
                else:
                    result = await asyncio.to_thread(fn, *args)

                w0, h0, poses_px = self.poses_px_from_result(result, img_shape)
                infer_ms = (time.perf_counter() - t0) * 1000.0
                self.stats["infer_ms_last"] = float(infer_ms)
                prev_avg = float(self.stats["infer_ms_avg"])
                self.stats["infer_ms_avg"] = prev_avg * 0.9 + infer_ms * 0.1

                num_poses = len(poses_px)
                pts_per_pose = [len(p) for p in poses_px]
                self._dbg(f"POSES count={num_poses} pts_per_pose={pts_per_pose} infer_ms={infer_ms:.2f}")

                changed = (poses_px != self.last_poses_px)
                if changed or self.last_change_ms == 0:
                    self.last_change_ms = ts_ms

                if (ts_ms - self.last_sent_ms) > IDLE_TO_FORCE_KF_MS or \
                   (ts_ms - self.last_change_ms) > IDLE_TO_FORCE_KF_MS:
                    self.idle_start_ms = self.idle_start_ms or ts_ms
                else:
                    self.idle_start_ms = None

                external_kf = self.need_keyframe
                self.need_keyframe = False

                force_key   = (self.last_poses_px is None) or (len(self.last_poses_px) != len(poses_px))
                gap_key     = (ts_ms - self.last_sent_ms) > 250
                stale_key   = (ts_ms - self.last_key_ms) >= 300
                nochange_kf = (ts_ms - self.last_change_ms) >= 400
                first_move_after_idle = changed and (self.idle_start_ms is not None)
                heartbeat_abs = ABSOLUTE_INTERVAL_MS > 0 and (ts_ms - self.last_abs_ms) >= ABSOLUTE_INTERVAL_MS

                need_key = (
                    external_kf or force_key or gap_key or stale_key or
                    nochange_kf or first_move_after_idle or heartbeat_abs
                )

                self.seq = (self.seq + 1) & 0xFFFF

                if "pack_pose_frame_delta" in globals():
                    packet = pack_pose_frame_delta(
                        self.last_poses_px, poses_px, w0, h0, need_key, seq=self.seq, ver=2
                    )
                    if need_key:
                        self.last_key_ms = ts_ms
                        self.last_abs_ms = ts_ms
                    packet_kind = "PD"
                else:
                    packet = pack_pose_frame(w0, h0, poses_px)
                    self.last_key_ms = ts_ms
                    self.last_abs_ms = ts_ms
                    need_key = True
                    packet_kind = "PO"

                if (self.results_dc and
                    self.results_dc.get_property("ready-state") == GstWebRTC.WebRTCDataChannelState.OPEN and
                    (self.results_dc.get_property("buffered-amount") or 0) < SEND_THRESHOLD):
                    try:
                        gbytes = GLib.Bytes(packet)
                        self.results_dc.emit("send-data", gbytes)
                    except Exception:
                        gstbuf = Gst.Buffer.new_allocate(None, len(packet), None)
                        gstbuf.fill(0, packet)
                        self.results_dc.emit("send-data", gstbuf)

                    self.last_sent_ms = ts_ms
                    self.last_poses_px = poses_px
                    self.stats["frames_sent"] = int(self.stats["frames_sent"]) + 1
                    self.stats["bytes_sent"]  = int(self.stats["bytes_sent"]) + len(packet)
                    if need_key:
                        self.stats["kf_sent"] = int(self.stats["kf_sent"]) + 1
                    else:
                        self.stats["delta_sent"] = int(self.stats["delta_sent"]) + 1

                    self._info(
                        f"PACKET kind={packet_kind} {'KF' if need_key else 'Δ'} "
                        f"seq={self.seq:04d} bytes={len(packet)} "
                        f"bufAmt={(self.results_dc.get_property('buffered-amount') or 0)} "
                        f"poses={num_poses} pts={pts_per_pose} "
                        f"infer_ms(last/avg)={self.stats['infer_ms_last']:.2f}/{self.stats['infer_ms_avg']:.2f}"
                    )

                    if RESULTS_REQUIRE_ACK:
                        self._awaiting_ack[self.seq] = ts_ms
                        self._dbg(f"Awaiting ACK for seq={self.seq}")
                else:
                    self._dbg("Skip send: DC closed or buffered-amount high")
            except Exception as e:
                self._warn(f"Inference/send error: {e}")
                continue

    # ───── Diagnostics snapshot ─────
    def snapshot(self) -> Dict[str, object]:
        try:
            pipe_state = None
            conn_state = None
            ice_state  = None
            sig_state  = None
            if self.pipeline:
                try:
                    st = self.pipeline.get_state(0.0)[1]
                    pipe_state = st.value_nick if st is not None else None
                except Exception:
                    pipe_state = None
            if self.webrtc:
                try:
                    conn_state = self.webrtc.get_property("connection-state").value_nick
                except Exception:
                    pass
                try:
                    ice_state = self.webrtc.get_property("ice-gathering-state").value_nick
                except Exception:
                    pass
                try:
                    sig_state = self.webrtc.get_property("signaling-state").value_nick
                except Exception:
                    pass
            res_state = (self.results_dc.get_property("ready-state").value_nick
                         if self.results_dc else None)
            ctrl_state = (self.ctrl_dc.get_property("ready-state").value_nick
                          if self.ctrl_dc else None)

            factories = {
                "webrtcbin": _has_factory("webrtcbin"),
                "rtph264depay": _has_factory("rtph264depay"),
                "rtph265depay": _has_factory("rtph265depay"),
                "rtpav1depay": _has_factory("rtpav1depay"),
                "h264parse": _has_factory("h264parse"),
                "h265parse": _has_factory("h265parse"),
                "avdec_h264": _has_factory("avdec_h264"),
                "avdec_h265": _has_factory("avdec_h265"),
                "vah265dec": _has_factory("vah265dec"),
                "vaapih265dec": _has_factory("vaapih265dec"),
                "nvh265dec": _has_factory("nvh265dec"),
                "av1dec": _has_factory("av1dec"),
                "dav1dec": _has_factory("dav1dec"),
                "vapostproc": _has_factory("vapostproc"),
                "vaapipostproc": _has_factory("vaapipostproc"),
                "nvvideoconvert": _has_factory("nvvideoconvert"),
                "nvvidconv": _has_factory("nvvidconv"),
                "videoconvert": _has_factory("videoconvert"),
                "appsink": _has_factory("appsink"),
            }

            return {
                "sid": self.sid,
                "pipeline_state": pipe_state,
                "connection_state": conn_state,
                "signaling_state": sig_state,
                "ice_gathering_state": ice_state,
                "results_dc_state": res_state,
                "ctrl_dc_state": ctrl_state,
                "stats": dict(self.stats),
                "bus_tail": list(self._bus_tail),
                "factories": factories,
            }
        except Exception as e:
            return {"snapshot_error": str(e)}

# ─────────────── Constructor del Blueprint WebRTC (Sanic) ───────────────
def build_webrtc_blueprint(
    *,
    make_mp_image: Callable,
    detect_image: Callable,
    detect_video: Callable,
    poses_px_from_result: Callable,
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
        sess = GSTWebRTCSession(
            make_mp_image=make_mp_image,
            detect_image=detect_image,
            detect_video=detect_video,
            poses_px_from_result=poses_px_from_result,
            loop=loop,
        )
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
