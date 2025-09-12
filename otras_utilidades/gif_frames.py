# gif_frames.py
# pip install imageio pillow

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple

# ── CONFIG ────────────────────────────────────────────────────────────────────
# INPUT_PATH can be a folder OR a single .gif file
INPUT_PATH  = Path(r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE\TESIS\CODIGO\validador_retratos_webrtc\cuts\GIF_GIROS")  # ← change this
OUTPUT_ROOT = Path(r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE\TESIS\CODIGO\validador_retratos_webrtc\cuts\FRAMES_GIROS")           # ← change this
RECURSIVE   = True   # search subfolders if INPUT_PATH is a folder
SAVE_METADATA = True # write _frames_metadata.json per GIF
# ──────────────────────────────────────────────────────────────────────────────

# Preferred backend: imageio (robust GIF disposal/partial-frame handling)
try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

# Fallback: Pillow
try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def extract_with_imageio(gif_path: Path, out_dir: Path) -> Dict[str, Any]:
    """Extract frames using imageio. Returns {'n_frames': int, 'durations_ms': list|None}."""
    meta = iio.immeta(gif_path)  # global-ish metadata; may contain duration(s)
    frames = iio.imread(gif_path, index=None)  # (n, h, w, c) uint8 or (h,w,c) if 1 frame
    if frames.ndim == 3:  # single frame
        frames = frames[None, ...]
    n = frames.shape[0]

    durations = None
    if isinstance(meta, dict):
        d = meta.get("duration", None)  # may be list or scalar (ms)
        if isinstance(d, list):
            durations = d
        elif isinstance(d, (int, float)):
            durations = [int(d)] * n

    for i in range(n):
        out_path = out_dir / f"frame_{i:04d}.png"
        iio.imwrite(out_path, frames[i])

    return {"n_frames": n, "durations_ms": durations}


def extract_with_pillow(gif_path: Path, out_dir: Path) -> Dict[str, Any]:
    """
    Extract frames using Pillow. Composes onto a persistent RGBA canvas.
    Returns {'n_frames': int, 'durations_ms': list}.
    """
    if not HAS_PIL:
        raise RuntimeError("Pillow not installed, and imageio unavailable.")

    im = Image.open(gif_path)
    n = getattr(im, "n_frames", 1)
    durations: List[int] = []

    canvas = Image.new("RGBA", im.size, (0, 0, 0, 0))
    palette = im.getpalette()

    for i in range(n):
        im.seek(i)
        if im.mode == "P" and palette:
            im.putpalette(palette)

        frame = im.convert("RGBA")
        x = im.info.get("x", 0)
        y = im.info.get("y", 0)

        # Disposal=2 (restore to background): clear area before pasting
        if im.info.get("disposal", 0) == 2:
            clear = Image.new("RGBA", frame.size, (0, 0, 0, 0))
            canvas.paste(clear, (x, y))

        canvas.paste(frame, (x, y), frame)
        (out_dir / f"frame_{i:04d}.png").write_bytes(
            canvas_to_bytes(canvas)
        )

        durations.append(int(im.info.get("duration", 0)))

    return {"n_frames": n, "durations_ms": durations}


def canvas_to_bytes(img_rgba: Image.Image) -> bytes:
    """Save a PIL Image to PNG bytes (small helper to avoid reopening files)."""
    from io import BytesIO
    buf = BytesIO()
    img_rgba.save(buf, format="PNG")
    return buf.getvalue()


def write_metadata_json(out_dir: Path, meta: Dict[str, Any]) -> None:
    import json
    with (out_dir / "_frames_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def extract_gif(gif_path: Path, dest_root: Path, save_meta: bool) -> Tuple[Path, Dict[str, Any]]:
    subdir = dest_root / gif_path.stem
    safe_mkdir(subdir)

    if HAS_IMAGEIO:
        meta = extract_with_imageio(gif_path, subdir)
    elif HAS_PIL:
        meta = extract_with_pillow(gif_path, subdir)
    else:
        raise RuntimeError("Install at least one backend: pip install imageio pillow")

    meta_out = {"gif": str(gif_path), "out_dir": str(subdir), **meta}
    if save_meta:
        write_metadata_json(subdir, meta_out)
    return subdir, meta_out


def find_gifs(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".gif" else []
    if recursive:
        return sorted(p for p in input_path.rglob("*.gif") if p.is_file())
    return sorted(p for p in input_path.glob("*.gif") if p.is_file())


def run() -> None:
    input_path = INPUT_PATH.resolve()
    output_root = OUTPUT_ROOT.resolve()
    safe_mkdir(output_root)

    gifs = find_gifs(input_path, RECURSIVE)
    if not gifs:
        print("No GIF files found.")
        return

    print(f"Found {len(gifs)} GIF(s). Extracting to '{output_root}'...")
    for gif in gifs:
        rel = gif.name if input_path.is_file() else gif.relative_to(input_path)
        print(f" - {rel}")
        out_dir, meta = extract_gif(gif, output_root, save_meta=SAVE_METADATA)
        print(f"   -> {meta['n_frames']} frame(s) -> {out_dir}")

    print("Done.")


if __name__ == "__main__":
    run()
