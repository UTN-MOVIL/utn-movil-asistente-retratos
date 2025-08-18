# robust_bytes.py (or inline near your DC handlers)
def _as_bytes(obj):
    if obj is None:
        return b""
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return bytes(obj)
    try:
        from gi.repository import GLib, Gst
        # GLib.Bytes -> bytes
        if isinstance(obj, GLib.Bytes):
            try:
                mv = obj.get_data()           # often a memoryview
                return mv.tobytes() if isinstance(mv, memoryview) else bytes(mv)
            except Exception:
                # some GI builds let bytes(obj) work
                return bytes(obj)
        # Gst.Buffer -> bytes
        if isinstance(obj, Gst.Buffer):
            ok, mapinfo = obj.map(Gst.MapFlags.READ)
            try:
                if ok:
                    data = mapinfo.data
                    return data.tobytes() if isinstance(data, memoryview) else bytes(data)
            finally:
                if ok:
                    obj.unmap(mapinfo)
    except Exception:
        pass
    # last resort
    return bytes(obj) if hasattr(obj, "__bytes__") else b""
