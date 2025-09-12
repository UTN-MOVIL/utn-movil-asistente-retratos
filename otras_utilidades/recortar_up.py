from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, List, Optional

# ── CONFIG ───────────────────────────────────────────────────────────────
FOLDER   = Path(r"C:\Users\Administrador\Desktop\FINAL")  # ← change this
PIXELS   = 32                               # how many rows to cut from the bottom
RECURSIVE = True                           # process subfolders too
EXTS: Optional[Iterable[str]] = None       # e.g. {"jpg","png"} or None for common formats
# ─────────────────────────────────────────────────────────────────────────

# Paste your crop_bottom_inplace() here, or import it:
# from your_module import crop_bottom_inplace
from PIL import Image
import tempfile, os

def crop_bottom_inplace(image_path: str | Path, pixels: int) -> Tuple[int, int]:
    """
    Crop `pixels` rows from the bottom of the image and overwrite the same file.

    Args:
        image_path: Path to the image (will be overwritten).
        pixels: Non-negative number of pixels to remove from the bottom.

    Returns:
        (new_width, new_height)

    Raises:
        FileNotFoundError: if the file doesn't exist.
        ValueError: if pixels < 0 or pixels >= image height.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if pixels < 0:
        raise ValueError("`pixels` must be >= 0")

    with Image.open(path) as im:
        w, h = im.size
        if pixels == 0:
            return (w, h)
        if pixels >= h:
            raise ValueError(f"Cannot crop {pixels} px from height {h}.")

        # Preserve metadata when possible
        exif = im.info.get("exif")
        icc  = im.info.get("icc_profile")

        cropped = im.crop((0, 0, w, h - pixels))

        # Prepare atomic save to a temp file in the same folder
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=path.stem + "_tmp_", suffix=path.suffix, dir=str(path.parent)
        )
        os.close(tmp_fd)

        save_kwargs = {}
        if exif: save_kwargs["exif"] = exif
        if icc:  save_kwargs["icc_profile"] = icc

        # For JPEGs, try to keep quality/subsampling settings (Pillow ≥ 9.1)
        try:
            if (im.format or "").upper() == "JPEG" or path.suffix.lower() in (".jpg", ".jpeg"):
                save_kwargs.update(dict(quality="keep", subsampling="keep", optimize=True))
        except TypeError:
            # Older Pillow: silently ignore 'keep'
            pass

        cropped.save(tmp_name, **save_kwargs)
        os.replace(tmp_name, path)  # atomic on POSIX & Windows

        return (w, h - pixels)

def crop_bottom_inplace(image_path: str | Path, pixels: int) -> Tuple[int, int]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if pixels < 0:
        raise ValueError("`pixels` must be >= 0")

    with Image.open(path) as im:
        w, h = im.size
        if pixels == 0:
            return (w, h)
        if pixels >= h:
            raise ValueError(f"Cannot crop {pixels} px from height {h}.")

        exif = im.info.get("exif")
        icc  = im.info.get("icc_profile")
        cropped = im.crop((0, 0, w, h - pixels))

        fd, tmp_name = tempfile.mkstemp(
            prefix=path.stem + "_tmp_", suffix=path.suffix, dir=str(path.parent)
        )
        os.close(fd)

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
        return (w, h - pixels)


def batch_crop_bottom_inplace(
    folder: str | Path,
    pixels: int,
    *,
    recursive: bool = True,
    exts: Optional[Iterable[str]] = None,
) -> List[Tuple[Path, Tuple[int, int]]]:
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    if exts is None:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    else:
        exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}

    paths = (folder.rglob("*") if recursive else folder.glob("*"))
    results: List[Tuple[Path, Tuple[int, int]]] = []

    for p in paths:
        if p.is_file() and p.suffix.lower() in exts:
            try:
                new_size = crop_bottom_inplace(p, pixels)
                results.append((p, new_size))
                print(f"OK  {p} -> {new_size}")
            except Exception as e:
                print(f"SKIP {p} ({type(e).__name__}: {e})")

    print(f"Done. Processed {len(results)} file(s).")
    return results


if __name__ == "__main__":
    batch_crop_bottom_inplace(
        FOLDER,
        PIXELS,
        recursive=RECURSIVE,
        exts=EXTS,
    )
