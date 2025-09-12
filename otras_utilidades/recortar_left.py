from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, List, Tuple
import shutil, os, tempfile
from PIL import Image

# ── CONFIG ───────────────────────────────────────────────────────────────
SRC_DIR = Path(r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE\TESIS\CODIGO\validador_retratos_webrtc\cuts\RESULT")      # ← change this
DST_DIR = Path(r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE\TESIS\CODIGO\validador_retratos_webrtc\cuts\FINAL") # ← change this
RECURSIVE = True                                # process subfolders too
EXTS: Optional[Iterable[str]] = None            # e.g. {"jpg","png"} or None for common formats
# ─────────────────────────────────────────────────────────────────────────

def crop_right_excess_inplace(image_path: str | Path) -> Tuple[int, int]:
    """
    If image width > 336, remove (width - 336) columns from the RIGHT edge and overwrite the same file.
    Returns (new_width, new_height). If width <= 336, leaves file unchanged and returns original size.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)

    with Image.open(path) as im:
        w, h = im.size
        diff = w - 336
        if diff <= 0:
            return (w, h)

        exif = im.info.get("exif")
        icc  = im.info.get("icc_profile")

        cropped = im.crop((diff, 0, w, h))  # keep RIGHT portion (crop diff px from LEFT edge)

        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=path.stem + "_tmp_", suffix=path.suffix, dir=str(path.parent)
        )
        os.close(tmp_fd)

        save_kwargs = {}
        if exif: save_kwargs["exif"] = exif
        if icc:  save_kwargs["icc_profile"] = icc
        try:
            if (im.format or "").upper() == "JPEG" or path.suffix.lower() in (".jpg", ".jpeg"):
                save_kwargs.update(dict(quality="keep", subsampling="keep", optimize=True))
        except TypeError:
            pass

        cropped.save(tmp_name, **save_kwargs)
        os.replace(tmp_name, path)
        return (w - diff, h)

def _normalize_exts(exts: Optional[Iterable[str]]) -> set[str]:
    if exts is None:
        return {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    return {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}

def batch_run_crop_right(
    src_dir: str | Path,
    dst_dir: str | Path,
    *,
    recursive: bool = True,
    exts: Optional[Iterable[str]] = None,
) -> List[Tuple[Path, Tuple[int, int]]]:
    """Copy images from src→dst (preserving structure) and crop the copies in place."""
    src = Path(src_dir)
    dst = Path(dst_dir)
    if not src.is_dir():
        raise NotADirectoryError(src)
    same_dir = src.resolve() == dst.resolve()

    inc_exts = _normalize_exts(exts)
    it = src.rglob("*") if recursive else src.glob("*")

    results: List[Tuple[Path, Tuple[int, int]]] = []
    for p in it:
        if p.is_file() and p.suffix.lower() in inc_exts:
            try:
                if same_dir:
                    # In-place cropping
                    new_size = crop_right_excess_inplace(p)
                    results.append((p, new_size))
                    print(f"OK  {p} -> {new_size}")
                else:
                    # Mirror relative path under destination
                    rel = p.relative_to(src)
                    out_path = dst / rel
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    # Copy then crop the copy in place
                    shutil.copy2(p, out_path)
                    new_size = crop_right_excess_inplace(out_path)
                    results.append((out_path, new_size))
                    print(f"OK  {out_path} (from {p}) -> {new_size}")
            except Exception as e:
                print(f"SKIP {p} ({type(e).__name__}: {e})")

    print(f"Done. Processed {len(results)} file(s).")
    return results

if __name__ == "__main__":
    batch_run_crop_right(SRC_DIR, DST_DIR, recursive=RECURSIVE, exts=EXTS)
