# connection/decoding.py — RTP video decode chain builder for GStreamer
# Builds a Gst.Bin that takes application/x-rtp (video) on its "sink" pad and
# outputs CPU RGB frames to an appsink (emitting "new-sample").
#
# Usage:
#   bin_, appsink = build_rtp_video_decode_bin(encoding, on_new_sample, dbg=..., warn=...)
#   pipeline.add(bin_)
#   src_pad.link(bin_.get_static_pad("sink"))
#   bin_.sync_state_with_parent()

from __future__ import annotations
import contextlib
import os
from typing import Callable, Optional, Tuple

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp

# ── Optional dependency on your codec registry (with safe fallback)
try:
    from .codec import CANDIDATE_DECODERS, _find_first_factory
except Exception:
    # H.265 default candidates (HW first, then software)
    CANDIDATE_DECODERS = ["vah265dec", "vaapih265dec", "nvh265dec", "avdec_h265"]

    def _find_first_factory(names):
        for n in names:
            if Gst.ElementFactory.find(n):
                return n
        return None

# Prefer VA-API for H.264 (fallback to NV then software)
CANDIDATE_DECODERS_H264 = ["vah264dec", "vaapih264dec", "nvh264dec", "avdec_h264"]


def _has_factory(name: str) -> bool:
    return Gst.ElementFactory.find(name) is not None


# ──────────────────────── HW detection helpers + logging ───────────────────────

def _is_va_factory(factory_name: str) -> bool:
    # Covers gstreamer-va (va*) and gstreamer-vaapi (vaapi*)
    n = (factory_name or "").lower()
    return n.startswith("va") or "vaapi" in n or n.startswith("vah")

def _is_nv_factory(factory_name: str) -> bool:
    n = (factory_name or "").lower()
    return n.startswith("nv") or "nvh" in n or "nvcodec" in n

def _get_prop(elem: Gst.Element, name: str):
    with contextlib.suppress(Exception):
        return elem.get_property(name)
    return None

def _describe_gpu_from_drm_device(dev_path: str) -> Optional[str]:
    """
    Try to resolve a DRM render node like /dev/dri/renderD128 to a vendor name by
    reading /sys/class/drm/<node>/device/vendor (0x1002=AMD, 0x8086=Intel, 0x10de=NVIDIA).
    """
    try:
        real = os.path.realpath(str(dev_path))
        node = os.path.basename(real)  # e.g., renderD128
        base = f"/sys/class/drm/{node}/device"
        vendor_path = os.path.join(base, "vendor")
        device_path = os.path.join(base, "device")
        if os.path.exists(vendor_path):
            vendor_id = open(vendor_path, "r").read().strip().lower()
            pci_id = open(device_path, "r").read().strip().lower() if os.path.exists(device_path) else "unknown"
            vendor_map = {"0x1002": "AMD", "0x8086": "Intel", "0x10de": "NVIDIA"}
            vendor = vendor_map.get(vendor_id, vendor_id)
            return f"{vendor} (PCI {pci_id}) via {real}"
    except Exception:
        pass
    return None

def _log_decoder_hw_details(dec: Gst.Element, enc: str, dbg: Callable[[str], None] | None):
    if not dbg or not dec:
        return
    try:
        factory = dec.get_factory().get_name() if dec.get_factory() else "unknown"
    except Exception:
        factory = "unknown"

    # Base message about chosen decoder
    msg = [f"Decoder selected for {enc}: '{factory}'"]

    # Determine HW vs SW path
    if _is_va_factory(factory):
        msg.append("(HW: VA-API)")
        # Try to fetch DRM/render device path
        drm_dev = _get_prop(dec, "drm-device") or _get_prop(dec, "device") \
                  or _get_prop(dec, "render-device") or _get_prop(dec, "device-path")
        if drm_dev:
            gpu_desc = _describe_gpu_from_drm_device(str(drm_dev))
            if gpu_desc:
                msg.append(f"→ Using DRM node: {gpu_desc}")
            else:
                msg.append(f"→ DRM device: {drm_dev}")
        # Some VA decoders expose a 'display' or 'va-display'
        va_display = _get_prop(dec, "display") or _get_prop(dec, "va-display")
        if va_display:
            msg.append(f"(display={va_display})")
        dbg(" ".join(msg))
    elif _is_nv_factory(factory):
        msg.append("(HW: NVDEC)")
        gpu_id = _get_prop(dec, "gpu-id")
        if gpu_id is not None:
            msg.append(f"(gpu-id={gpu_id})")
        dbg(" ".join(msg))
    else:
        msg.append("(SW decode)")
        dbg(" ".join(msg))


# ──────────────────────── Low-latency decoder tweaks ───────────────────────

def _apply_decoder_latency_tweaks(dec: Gst.Element, dbg: Callable[[str], None] | None = None) -> None:
    """Best-effort low-latency settings depending on the decoder in use.
    - NVDEC (nvh264dec/nvh265dec): disable-dpb=true
    - Jetson (nvv4l2decoder):      low-latency=true
    - FFmpeg (avdec_*):            thread-type=slice and threads=2..4
    Applies only if the property exists on the element (safe/no-op otherwise)."""
    if not dec:
        return

    # Helper: set a property if supported (avoid noisy warnings)
    def _set_prop(name: str, value) -> bool:
        with contextlib.suppress(Exception):
            # Some builds expose hyphen or underscore variants; try both.
            for candidate in (name, name.replace('-', '_')):
                if hasattr(dec, 'find_property') and dec.find_property(candidate):
                    dec.set_property(candidate, value)
                    if dbg:
                        try:
                            factory = dec.get_factory().get_name()
                        except Exception:
                            factory = 'unknown'
                        dbg(f"Decoder tweak: {factory}.{candidate}={value}")
                    return True
        return False

    try:
        factory = dec.get_factory().get_name().lower() if dec.get_factory() else ''
    except Exception:
        factory = ''

    # NVDEC (discrete NVIDIA via nvcodec)
    if factory in ("nvh264dec", "nvh265dec"):
        _set_prop("disable-dpb", True)

    # Jetson V4L2 decoder
    if factory == "nvv4l2decoder":
        _set_prop("low-latency", True)

    # FFmpeg software decoders (avdec_*)
    if factory.startswith("avdec_"):
        # Prefer slice threading to reduce per-frame latency
        ok = _set_prop("thread-type", "slice")
        if not ok:
            # Some older builds may accept ints; slice is typically 1
            _set_prop("thread-type", 1)
        # Cap thread count between 2 and 4
        threads = max(2, min(4, (os.cpu_count() or 2)))
        _set_prop("threads", threads)


def _make_depay_and_parse(encoding: str) -> Tuple[Optional[Gst.Element], Optional[Gst.Element]]:
    enc = (encoding or "").upper()
    depay = parse = None
    if enc in ("VP8",):
        depay = Gst.ElementFactory.make("rtpvp8depay", None)
    elif enc in ("VP9",):
        depay = Gst.ElementFactory.make("rtpvp9depay", None)
    elif enc in ("H264", "H264-SVC"):
        depay = Gst.ElementFactory.make("rtph264depay", None)
        parse = Gst.ElementFactory.make("h264parse", None)
        if parse:
            # Make VA/NV decoders happier: push SPS/PPS periodically, AU aligned, byte-stream
            with contextlib.suppress(Exception):
                parse.set_property("config-interval", 1)
            with contextlib.suppress(Exception):
                parse.set_property("alignment", "au")
            with contextlib.suppress(Exception):
                parse.set_property("stream-format", "byte-stream")
    elif enc in ("H265", "HEVC", "H265/90000"):
        depay = Gst.ElementFactory.make("rtph265depay", None)
        parse = Gst.ElementFactory.make("h265parse", None)
        if parse:
            with contextlib.suppress(Exception):
                parse.set_property("config-interval", 1)
            with contextlib.suppress(Exception):
                parse.set_property("alignment", "au")
            with contextlib.suppress(Exception):
                parse.set_property("stream-format", "byte-stream")
    elif "AV1" in enc:
        depay = Gst.ElementFactory.make("rtpav1depay", None)
        # av1parse not strictly required post-depay
    return depay, parse


def _make_decoder_for(
    encoding: str,
    dbg: Callable[[str], None] | None = None,
    warn: Callable[[str], None] | None = None
) -> Optional[Gst.Element]:
    enc = (encoding or "").upper()
    dec = None
    if enc in ("VP8",):
        dec = Gst.ElementFactory.make("vp8dec", None)
    elif enc in ("VP9",):
        dec = Gst.ElementFactory.make("vp9dec", None)
    elif enc in ("H264", "H264-SVC"):
        name = _find_first_factory(CANDIDATE_DECODERS_H264) or "avdec_h264"
        dec = Gst.ElementFactory.make(name, None)
    elif enc in ("H265", "HEVC", "H265/90000"):
        name = _find_first_factory(CANDIDATE_DECODERS) or "avdec_h265"
        dec = Gst.ElementFactory.make(name, None)
    elif "AV1" in enc:
        dec = (Gst.ElementFactory.make("av1dec", None)
               or Gst.ElementFactory.make("dav1dec", None))
    if dec:
        _apply_decoder_latency_tweaks(dec, dbg)
    if dbg and dec:
        _log_decoder_hw_details(dec, enc, dbg)
    if warn and not dec:
        warn(f"No decoder available for encoding '{enc}'")
    return dec


def _maybe_postproc_after(
    dec: Gst.Element,
    dbg: Callable[[str], None] | None = None
) -> Tuple[Optional[Gst.Element], Optional[Gst.Element]]:
    """
    If the decoder outputs GPU memory, insert a vendor postproc and a capsfilter
    to force download to system memory.
    Returns (postproc, caps_to_sysmem).
    """
    if not dec:
        return None, None
    name = ""
    try:
        name = dec.get_factory().get_name()
    except Exception:
        pass

    # Broaden detection to cover vah264/265 and NV variants
    use_va = name.startswith("va") or "vaapi" in name or name.startswith("vah")
    use_nv = name.startswith("nv") or "nvh" in name

    postproc = caps_to_sys = None
    if use_va:
        pp_name = ("vapostproc" if _has_factory("vapostproc")
                   else ("vaapipostproc" if _has_factory("vaapipostproc") else None))
        if pp_name:
            postproc = Gst.ElementFactory.make(pp_name, None)
            if dbg:
                dbg(f"Inserted VA-API postproc: {pp_name} (GPU surfaces → CPU)")
        caps_to_sys = Gst.ElementFactory.make("capsfilter", "caps_to_sysmem")
        # Download VA surfaces to system memory with a defined format
        caps_to_sys.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw,format=NV12,memory:SystemMemory")
        )
        if dbg:
            dbg("Forcing download of VA surfaces: caps=video/x-raw,format=NV12,memory:SystemMemory")
    elif use_nv:
        pp_name = ("nvvideoconvert" if _has_factory("nvvideoconvert")
                   else ("nvvidconv" if _has_factory("nvvidconv") else None))
        if pp_name:
            postproc = Gst.ElementFactory.make(pp_name, None)
            if dbg:
                dbg(f"Inserted NV postproc: {pp_name} (NVMM → CPU)")
        caps_to_sys = Gst.ElementFactory.make("capsfilter", "caps_to_sysmem")
        # Dropping NVMM memory type is enough; format can be negotiated
        caps_to_sys.set_property("caps", Gst.Caps.from_string("video/x-raw"))
        if dbg:
            dbg("Forcing download of NVMM memory: caps=video/x-raw")

    return postproc, caps_to_sys


def build_rtp_video_decode_bin(
    encoding_name: str,
    on_new_sample: Callable[[GstApp.AppSink], Gst.FlowReturn],
    *,
    want_rgb: bool = True,
    dbg: Callable[[str], None] | None = None,
    warn: Callable[[str], None] | None = None,
    name: str = "rxdecbin",
) -> Tuple[Gst.Bin, GstApp.AppSink]:
    """
    Creates a Gst.Bin with a *ghost sink pad* that accepts application/x-rtp (video)
    and ends in an appsink (CPU memory, RGB if want_rgb=True).
    Returns (bin, appsink). Caller must add to pipeline and link the src pad → bin.sink.
    """
    bin_ = Gst.Bin.new(name)
    if bin_ is None:
        raise RuntimeError("Failed to create Gst.Bin")

    q_in = Gst.ElementFactory.make("queue", None)
    depay, parse = _make_depay_and_parse(encoding_name)
    dec = _make_decoder_for(encoding_name, dbg=dbg, warn=warn)
    if depay is None or dec is None:
        raise RuntimeError(f"No depay/decoder available for encoding '{encoding_name}'")

    postproc, caps_to_sys = _maybe_postproc_after(dec, dbg=dbg)

    swcvt = Gst.ElementFactory.make("videoconvert", "swcvt")  # ensures CPU colorspace
    # Enable QoS on the CPU converter (optional but recommended)
    with contextlib.suppress(Exception):
        swcvt.set_property("qos", True)

    caps_rgb = None
    if want_rgb:
        caps_rgb = Gst.ElementFactory.make("capsfilter", "caps_rgb")
        caps_rgb.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGB"))

    # Create the leaky queue right before appsink
    q2 = Gst.ElementFactory.make("queue", "leaky_to_sink")
    q2.set_property("leaky", 2)               # downstream leaky
    q2.set_property("max-size-buffers", 1)
    q2.set_property("max-size-time", 0)
    q2.set_property("max-size-bytes", 0)

    appsink: GstApp.AppSink = Gst.ElementFactory.make("appsink", "appsink")
    appsink.set_property("emit-signals", True)
    appsink.set_property("sync", False)
    appsink.set_property("max-buffers", 1)
    appsink.set_property("drop", True)
    appsink.connect("new-sample", on_new_sample)

    # Build the chain, inserting q2 just before appsink
    chain = [q_in, depay]
    if parse:
        chain.append(parse)
    chain.append(dec)
    if postproc:
        chain.append(postproc)
    if caps_to_sys:
        chain.append(caps_to_sys)
    if swcvt:
        chain.append(swcvt)
    if caps_rgb:
        chain.append(caps_rgb)
    chain.append(q2)
    chain.append(appsink)

    for e in chain:
        bin_.add(e)

    # Link elements inside the bin
    for a, b in zip(chain[:-1], chain[1:]):
        if not a.link(b) and warn:
            warn(f"LINK FAIL {a.name} -> {b.name}")

    # Expose a ghost "sink" pad on the bin (points to q_in.sink)
    sinkpad = q_in.get_static_pad("sink")
    ghost = Gst.GhostPad.new("sink", sinkpad)
    ghost.set_active(True)
    bin_.add_pad(ghost)

    if dbg:
        try:
            dec_name = dec.get_factory().get_name()
            dep_name = depay.get_factory().get_name()
            rgb_flag = bool(caps_rgb)
            dbg(f"Decode bin ready: depay='{dep_name}', decoder='{dec_name}', rgb={rgb_flag}")
            # Extra hint: if VA-API/NV decoder chosen, we're on GPU decode path
            if _is_va_factory(dec_name):
                dbg("GPU decode path active (VA-API). If using AMD, this is your AMD GPU via VA-API.")
            elif _is_nv_factory(dec_name):
                dbg("GPU decode path active (NVDEC).")
            else:
                dbg("Software decode path active.")
        except Exception:
            pass

    return bin_, appsink


def attach_rtp_video_decode_chain(
    pipeline: Gst.Pipeline,
    src_pad: Gst.Pad,
    encoding_name: str,
    on_new_sample: Callable[[GstApp.AppSink], Gst.FlowReturn],
    *,
    dbg: Callable[[str], None] | None = None,
    warn: Callable[[str], None] | None = None,
) -> Gst.Bin:
    """
    Convenience wrapper: builds the bin, adds it to the pipeline, links src_pad→bin.sink,
    and syncs it to the parent's state. Returns the created bin.
    """
    bin_, _appsink = build_rtp_video_decode_bin(
        encoding_name, on_new_sample, dbg=dbg, warn=warn
    )
    pipeline.add(bin_)
    # Link webrtcbin's newly-added src pad → our bin sink
    ret = src_pad.link(bin_.get_static_pad("sink"))
    if ret != Gst.PadLinkReturn.OK and warn:
        warn(f"Failed to link RTP pad to decode bin: {ret}")
    # Ensure states
    bin_.sync_state_with_parent()
    if dbg:
        dbg(f"Attached decode chain for '{encoding_name}' with result: {ret.name}")
    return bin_
