from __future__ import annotations
from pathlib import Path
from typing import Sequence, Tuple, List, Optional
import numpy as np
from PIL import Image

Point = Tuple[int, int]
Line  = Tuple[Point, Point]

def partition_image_by_lines(
    image_path: str | Path,
    lines: Sequence[Line],
    *,
    output_dir: str | Path | None = None,
    prefix: str = "part",
    trim: bool = True,
    margin: int = 0,
    order: str = "row-major",  # "row-major" | "auto" | "none"
    name_fmt: Optional[str] = None,  # e.g. "{prefix}_{i:02d}.png" or "{prefix}_r{row}_c{col}.png"
) -> List[Image.Image]:
    """
    Partition an image by infinite lines and return/save one RGBA image per region.
    'row-major' orders pieces top→bottom, left→right (ideal for grids).
    """
    image_path = Path(image_path)
    im = Image.open(image_path).convert("RGBA")
    arr = np.array(im)  # H x W x 4
    H, W = arr.shape[:2]

    # Pixel coordinate grids (x right, y down)
    X, Y = np.meshgrid(np.arange(W, dtype=np.int32),
                       np.arange(H, dtype=np.int32))

    # Build sign fields per line
    signs: List[np.ndarray] = []
    vcnt = hcnt = 0
    for ((x1, y1), (x2, y2)) in lines:
        dx, dy = (x2 - x1), (y2 - y1)
        if dx == 0 and dy == 0:
            raise ValueError(f"Degenerate line with identical points: ({x1},{y1}).")
        s = (X - x1) * dy - (Y - y1) * dx
        signs.append(s)
        if abs(dx) < abs(dy):  # vertical-ish
            vcnt += 1
        elif abs(dy) < abs(dx):  # horizontal-ish
            hcnt += 1

    # Start with whole image, iteratively split by each line
    regions: List[np.ndarray] = [np.ones((H, W), dtype=bool)]
    for s in signs:
        gt, lt = (s > 0), (s < 0)
        new_regions: List[np.ndarray] = []
        for mask in regions:
            A = mask & gt
            B = mask & lt
            if A.any(): new_regions.append(A)
            if B.any(): new_regions.append(B)
        regions = new_regions
        if not regions:
            break

    # Compute bbox/centroid per region on the ORIGINAL canvas (robust to trimming)
    metas = []
    for m in regions:
        ys, xs = np.where(m)
        if ys.size == 0:
            # Shouldn't happen (we filtered empties), but keep a guard
            metas.append(dict(mask=m, bbox=(0, 0, 0, 0), cx=0.0, cy=0.0))
            continue
        x0, y0 = xs.min(), ys.min()
        x1, y1 = xs.max(), ys.max()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        metas.append(dict(mask=m, bbox=(x0, y0, x1, y1), cx=cx, cy=cy))

    # Decide ordering
    idxs = list(range(len(metas)))
    if order in ("row-major", "auto") and metas:
        expected = (vcnt + 1) * (hcnt + 1) if (vcnt or hcnt) else len(metas)

        # Try grid ordering if the count matches a grid
        if expected == len(metas) and (vcnt or hcnt):
            n_rows, n_cols = (hcnt + 1 if hcnt else 1), (vcnt + 1 if vcnt else 1)
            # Sort by cy, then split into n_rows using the largest (n_rows-1) gaps
            cy = np.array([m["cy"] for m in metas])
            cx = np.array([m["cx"] for m in metas])
            order_y = np.argsort(cy)
            diffs = np.diff(cy[order_y])
            # positions where to break rows
            if n_rows > 1:
                cuts = sorted(np.argpartition(diffs, -(n_rows - 1))[-(n_rows - 1):].tolist())
            else:
                cuts = []
            row_groups = []
            start = 0
            for cut in cuts + [len(order_y) - 1]:
                row_groups.append(order_y[start:cut + 1])
                start = cut + 1
            if not row_groups:  # single row
                row_groups = [order_y]

            # within each row, sort by cx ascending
            idxs = []
            row_col = {}  # map index -> (row, col) for naming
            for r, g in enumerate(row_groups, start=1):
                g_sorted = sorted(g.tolist(), key=lambda k: cx[k])
                for c, k in enumerate(g_sorted, start=1):
                    idxs.append(k)
                    row_col[k] = (r, c)
        else:
            # Fallback: simple row-major by top-left of bbox
            idxs = sorted(
                idxs,
                key=lambda i: (metas[i]["bbox"][1], metas[i]["bbox"][0])  # y0, then x0
            )
            row_col = {}

    else:
        # "none": keep whatever internal order
        row_col = {}

    # Convert masks to images (with optional trim)
    def mask_to_img(mask: np.ndarray) -> Image.Image:
        out = np.zeros_like(arr)
        out[mask] = arr[mask]
        img = Image.fromarray(out, mode="RGBA")
        if not trim:
            return img
        alpha = np.asarray(img)[:, :, 3]
        ys, xs = np.where(alpha > 0)
        if ys.size == 0:
            return img
        y0 = max(0, ys.min() - margin)
        y1 = min(H, ys.max() + 1 + margin)
        x0 = max(0, xs.min() - margin)
        x1 = min(W, xs.max() + 1 + margin)
        return img.crop((x0, y0, x1, y1))

    parts: List[Image.Image] = [mask_to_img(metas[i]["mask"]) for i in idxs]

    # Default naming template
    if name_fmt is None:
        # If we detected grid row/col, include them in the name
        if row_col:
            name_fmt = "{prefix}_r{row}_c{col}.png"
        else:
            name_fmt = "{prefix}_{i:02d}.png"

    # Optional save with ordered, human-friendly names
    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        for rank, i in enumerate(idxs, start=1):
            x0, y0, x1, y1 = metas[i]["bbox"]
            cx, cy = metas[i]["cx"], metas[i]["cy"]
            row, col = row_col.get(i, (None, None))
            fname = name_fmt.format(
                prefix=prefix, i=rank,
                row=row if row is not None else 1,
                col=col if col is not None else rank,
                x0=x0, y0=y0, x1=x1, y1=y1, cx=cx, cy=cy
            )
            parts[rank - 1].save(outdir / fname)

    return parts

# ───────────── Example ─────────────
if __name__ == "__main__":
    # Two vertical cuts at x=320 and x=320 → 3 parts: left, center, right
    imgs = partition_image_by_lines(
        r"C:\Users\Administrador\Downloads\ChatGPT Image 28 ago 2025, 00_19_09 (1) (3).png",
        [((316, 0), (316, 1599)), ((652, 0), (652, 1599)), ((988, 0), (988, 1599)), ((1324, 0), (1324, 1599)),
        ((0, 352), (1599, 352)), ((0, 693), (1599, 693))
        #, ((0, 988), (1599, 988))
        ],
        output_dir="cuts",
        prefix="frame",
        trim=True,
        margin=2,
        order="row-major",
        name_fmt="{prefix}_{i:04d}.png",          # 01..16 in reading order
        # or: name_fmt="{prefix}_r{row}_c{col}.png"  # r1_c1, r1_c2, ...
    )
    print(f"Generated {len(imgs)} parts")