#!/usr/bin/env python3
"""
Edge-refinement helper

This utility wraps several tricks that improve on “plain” Canny for
situations where edges are faint or defined mainly by colour:

    • Adaptive contrast boost (CLAHE) on the luminance channel
    • Customisable Canny thresholds
    • Channel-wise Canny on RGB  _and_  Lab a*/b* chroma channels
    • Morphological closing to bridge small gaps in the edge map
    • Optional visualisation for quick tuning

Requires:
    pip install opencv-python numpy
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, List

import cv2
import numpy as np
import os

def refine_edges(
    img_bgr: np.ndarray,
    *,
    canny_lo: int = 40,
    canny_hi: int = 120,
    use_clahe: bool = True,
    clahe_grid: Tuple[int, int] = (8, 8),
    use_rgb: bool = True,
    use_lab: bool = True,
    close_kernel: int = 3,
    show: bool = False,
) -> np.ndarray:
    """
    Return a refined binary edge map.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR order (cv2.imread() default).
    canny_lo, canny_hi : int
        Thresholds for the Canny operator.  Lower them to catch fainter edges.
    use_clahe : bool
        If True, run Contrast-Limited Adaptive Histogram Equalisation on
        the grayscale luminance before Canny.
    clahe_grid : (int, int)
        Tile grid size for CLAHE (bigger ⇒ coarser contrast boost).
    use_rgb, use_lab : bool
        Run Canny on individual RGB and/or Lab a*/b* channels and OR
        the results with the base grayscale edge map.
    close_kernel : int
        Size (pixels) of the square structuring element for a morphological
        closing pass (0 disables it).
    show : bool
        Pop up OpenCV windows to preview the intermediate steps.

    Returns
    -------
    edges : np.ndarray
        Single-channel uint8 image (255 = edge, 0 = background).
    """
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be a colour image in BGR format")

    # ── 1. Base grayscale + optional CLAHE ────────────────────────────────────
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=clahe_grid)
        gray = clahe.apply(gray)

    base_edges = cv2.Canny(gray, canny_lo, canny_hi)

    # ── 2. Optional per-channel Canny on colour spaces ───────────────────────
    extras: List[np.ndarray] = []

    if use_rgb:
        for ch in cv2.split(img_bgr):               # B, G, R order
            extras.append(cv2.Canny(ch, canny_lo, canny_hi))

    if use_lab:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        a_ch, b_ch = cv2.split(lab)[1:]             # skip L
        extras.append(cv2.Canny(a_ch, canny_lo, canny_hi))
        extras.append(cv2.Canny(b_ch, canny_lo, canny_hi))

    # ── 3. Combine all edge maps by OR union ─────────────────────────────────
    edges = base_edges.copy()
    for e in extras:
        edges = cv2.bitwise_or(edges, e)

    # ── 4. Optional morphological closing to fill tiny gaps ─────────────────
    if close_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)

    # ── 5. Debug view ────────────────────────────────────────────────────────
    if show:
        cv2.imshow("Refined edges (white = edge)", edges)
        if use_clahe:
            cv2.imshow("CLAHE grayscale", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return edges


# ─────────────────────────── Demo / CLI usage ───────────────────────────────
if __name__ == "__main__":
    img = cv2.imread("C:\\Users\\Administrador\\Documents\\INGENIERIA_EN_SOFTWARE\\TESIS\\CODIGO\\funcionalidades_validador_retratos\\tests\\test_image.jpg")
    edges = refine_edges(
        img
    )

    # Save alongside the original for quick inspection
    out_path = "_edges.png"
    cv2.imwrite(out_path, edges)
    print(f"[✓] Edge map written to {out_path}")
