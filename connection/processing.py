# connection/processing.py
from __future__ import annotations

import asyncio
import time
import inspect
from typing import Optional

import numpy as np
from gi.repository import Gst, GLib, GstWebRTC  # used by the original method

# Optional: for type checkers only (doesn't import at runtime)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .webrtc import GSTWebRTCSession


async def process_frames(session: "GSTWebRTCSession"):
    """
    Externalized version of GSTWebRTCSession._process_frames(session).
    We import webrtc lazily to avoid circular imports, and bind the
    globals/method alias so the pasted body can remain unchanged.
    """
    # Lazy import to avoid circular import at module load time
    from . import webrtc as W

    # Keep the original code using `self.` without edits:
    self = session  # alias so you can paste the original body as-is

    # Shadow the module-level constants the original method referenced:
    POSE_USE_VIDEO = W.POSE_USE_VIDEO
    ABSOLUTE_INTERVAL_MS = W.ABSOLUTE_INTERVAL_MS
    IDLE_TO_FORCE_KF_MS = W.IDLE_TO_FORCE_KF_MS
    RESULTS_REQUIRE_ACK = W.RESULTS_REQUIRE_ACK
    ACK_WARN_MS = W.ACK_WARN_MS

    # Packet helpers used in the method:
    pack_pose_frame = W.pack_pose_frame
    pack_pose_frame_delta = getattr(W, "pack_pose_frame_delta", None)

    # ─────────────────────────────────────────────────────────────
    # ⬇️ PASTE the original body of `_process_frames` here, UNCHANGED ⬇️
    # (keep everything from: `SEND_THRESHOLD = 32_768` down to the end)
    SEND_THRESHOLD = 32_768
    RECYCLE_AFTER_MS = 300
    MIN_SEND_MS = 33  # ~30 fps

    congested_since_ms: Optional[int] = None

    while True:
        try:
            (payload, w, h), pts_ns = await self.frame_q.get()

            # Accept RGBA as the default; tolerate RGB if upstream still sends it.
            # We infer the channel count from payload length to avoid mismatches.
            wh = int(w) * int(h)
            nbytes = len(payload)
            if nbytes == wh * 4:
                channels = 4  # RGBA
            elif nbytes == wh * 3:
                channels = 3  # RGB (fallback/compat)
            else:
                # Last-resort inference (e.g., if there’s padding) — will raise on reshape if invalid.
                channels = nbytes // wh
                if channels not in (3, 4):
                    raise ValueError(
                        f"Unexpected payload size={nbytes} for {w}x{h}; "
                        f"cannot infer channels (wanted 3 or 4)."
                    )

            frame = np.frombuffer(payload, dtype=np.uint8).reshape((h, w, channels))

            # Optional debug on first few frames to verify channel mode
            if self._proc_n < 5:
                self._dbg(f"PROCESS reshape: {w}x{h}x{channels} (bytes={nbytes})")

        except asyncio.CancelledError:
            break
        except Exception as e:
            self._warn(f"frame_q.get/reshape error: {e}")
            continue

        self._proc_n += 1
        if self._proc_n <= 5 or (self._proc_n % 30) == 0:
            dc_state = (self.results_dc.get_property("ready-state") if self.results_dc else None)
            self._dbg(f"PROCESS sample #{self._proc_n}: dequeued; dc_state={dc_state} q={self.frame_q.qsize()}")

        ts_ms = int(time.monotonic() * 1000)
        if ts_ms <= self.last_ts_input:
            ts_ms = self.last_ts_input + 1
        self.last_ts_input = ts_ms

        # ACK overdue checks (primary channel only)
        if RESULTS_REQUIRE_ACK and self._awaiting_ack:
            to_warn = [s for s, t0 in list(self._awaiting_ack.items()) if (ts_ms - t0) >= ACK_WARN_MS and s not in self._ack_warned]
            for s in to_warn:
                self._ack_warned.add(s)
                self._warn(f"ACK overdue for seq={s} > {ACK_WARN_MS}ms")

        # Rate-gate globally for all adapters
        if (ts_ms - self.last_sent_ms) < MIN_SEND_MS:
            continue

        try:
            async def run_one(ad):
                # If adapters provide a specialized RGBA constructor, prefer it.
                make_rgba = getattr(ad, "make_mp_image_rgba", None)
                if frame.shape[2] == 4 and callable(make_rgba):
                    mp_img = make_rgba(frame)  # RGBA path
                else:
                    mp_img = ad.make_mp_image(frame)  # existing path (should accept 3 or 4 channels)

                if POSE_USE_VIDEO:
                    if inspect.iscoroutinefunction(ad.detect_video):
                        res = await ad.detect_video(mp_img, ts_ms)
                    else:
                        res = await asyncio.to_thread(ad.detect_video, mp_img, ts_ms)
                else:
                    if inspect.iscoroutinefunction(ad.detect_image):
                        res = await ad.detect_image(mp_img)
                    else:
                        res = await asyncio.to_thread(ad.detect_image, mp_img)

                # Adapters should only need HxW; passing the whole shape (HxWxC) remains OK
                w0, h0, pts = ad.points_from_result(res, frame.shape)
                prev = self._prev_pts.get(ad.name)
                kf = prev is None or (prev is not None and len(prev) != len(pts))
                self.seq = (self.seq + 1) & 0xFFFF
                packet = (
                    pack_pose_frame_delta(prev, pts, w0, h0, keyframe=kf, seq=self.seq, ver=2)
                    if pack_pose_frame_delta is not None
                    else pack_pose_frame(w0, h0, pts)
                )
                return ad.name, (w0, h0), pts, packet, kf

            t0 = time.perf_counter()
            results = await asyncio.gather(*(run_one(ad) for ad in self.adapters))
            infer_ms = (time.perf_counter() - t0) * 1000.0
            self.stats["infer_ms_last"] = float(infer_ms)
            prev_avg = float(self.stats["infer_ms_avg"])
            self.stats["infer_ms_avg"] = prev_avg * 0.9 + infer_ms * 0.1

            primary_name = self.adapters[0].name
            primary_pts = next((pts for (name, _wh, pts, _pkt, _kf) in results if name == primary_name), None)
            changed = (primary_pts != self._prev_pts.get(primary_name))
            if changed or self.last_change_ms == 0:
                self.last_change_ms = ts_ms

            if (ts_ms - self.last_sent_ms) > IDLE_TO_FORCE_KF_MS or \
               (ts_ms - self.last_change_ms) > IDLE_TO_FORCE_KF_MS:
                self.idle_start_ms = self.idle_start_ms or ts_ms
            else:
                self.idle_start_ms = None

            external_kf = self.need_keyframe
            self.need_keyframe = False
            gap_key = (ts_ms - self.last_sent_ms) > 250
            stale_key = (ts_ms - self.last_key_ms) >= 300
            nochange_kf = (ts_ms - self.last_change_ms) >= 400
            first_move_after_idle = changed and (self.idle_start_ms is not None)
            heartbeat_abs = ABSOLUTE_INTERVAL_MS > 0 and (ts_ms - self.last_abs_ms) >= ABSOLUTE_INTERVAL_MS

            force_kf = (external_kf or gap_key or stale_key or nochange_kf or first_move_after_idle or heartbeat_abs)

            sent_any = False
            for name, (w0, h0), pts, packet, kf_local in results:
                dc = self.result_dcs.get(name)
                if not dc or dc.get_property("ready-state") != GstWebRTC.WebRTCDataChannelState.OPEN:
                    continue

                buf_amt = dc.get_property("buffered-amount") or 0
                if buf_amt >= SEND_THRESHOLD:
                    self.stats["drops_due_buffer"] = int(self.stats["drops_due_buffer"]) + 1
                    self._dbg(f"Skip '{name}' send: buffered-amount={buf_amt}")
                    continue

                if force_kf and pack_pose_frame_delta is not None:
                    packet = pack_pose_frame_delta(self._prev_pts.get(name), pts, w0, h0, keyframe=True, seq=self.seq, ver=2)
                    kf_local = True

                try:
                    try:
                        dc.emit("send-data", GLib.Bytes(packet))
                    except Exception:
                        gstbuf = Gst.Buffer.new_allocate(None, len(packet), None)
                        gstbuf.fill(0, packet)
                        dc.emit("send-data", gstbuf)

                    sent_any = True
                    self._prev_pts[name] = pts

                    self.stats["frames_sent"] = int(self.stats["frames_sent"]) + 1
                    self.stats["bytes_sent"] = int(self.stats["bytes_sent"]) + len(packet)
                    if kf_local:
                        self.stats["kf_sent"] = int(self.stats["kf_sent"]) + 1
                    else:
                        self.stats["delta_sent"] = int(self.stats["delta_sent"]) + 1

                    self._info(
                        f"[{name}] PACKET {'PD' if packet[:2]==b'PD' else 'PO'} "
                        f"{'KF' if kf_local or force_kf else 'Δ'} seq={self.seq:04d} "
                        f"bytes={len(packet)} bufAmt={(dc.get_property('buffered-amount') or 0)} "
                        f"pts-per-obj={[len(p) for p in pts]} "
                        f"infer_ms(last/avg)={self.stats['infer_ms_last']:.2f}/{self.stats['infer_ms_avg']:.2f}"
                    )
                except Exception as e:
                    self._warn(f"Send error on DC '{name}': {e}")
                    continue

            if sent_any:
                self.last_sent_ms = ts_ms
                if force_kf:
                    self.last_key_ms = ts_ms
                    self.last_abs_ms = ts_ms

                if RESULTS_REQUIRE_ACK and self.results_dc:
                    self._awaiting_ack[self.seq] = ts_ms
                    self._dbg(f"Awaiting ACK for seq={self.seq}")
            else:
                self._dbg("Skip send: all DCs closed or buffered-amount high")

        except Exception as e:
            self._warn(f"Inference/send error: {e}")
            continue
