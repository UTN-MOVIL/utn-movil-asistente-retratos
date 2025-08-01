#!/usr/bin/env python3
"""
Detect glasses *and* highlight the exact image region defined by
Face Mesh landmark indices [168, 6, 197, 195, 5, 4].

Dependencies
------------
pip install opencv-python mediapipe numpy
"""

from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Core routine
# ──────────────────────────────────────────────────────────────────────────
def detect_glasses(
    image_path: str,
    face_mesh_detector: "mp.solutions.face_mesh.FaceMesh",
    *,
    return_crop: bool = False,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Detect glasses and optionally return / display the landmark ROI."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = image.shape[:2]

    # Face-mesh inference
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh_detector.process(rgb)

    glasses_present = False
    crop: Optional[np.ndarray] = None

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        lm_px = np.array([(int(l.x * w), int(l.y * h)) for l in lm])

        idxs = np.array([168, 6, 197, 195, 5, 4])
        pts = lm_px[idxs]

        # Bounding box around the six landmarks
        x, y, ww, hh = cv2.boundingRect(pts)
        pad = 5

        # ── NEW: keep only the top N % of the box ──────────────────────────────
        FRACTION = 0.5                     # 0.5 → upper half, 0.3 → upper third …
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)

        # y1 is shifted downwards by FRACTION · hh instead of the full height
        y1 = min(h, y + int(hh * FRACTION) + pad)
        x1 = min(w, x + ww + pad)
        # ───────────────────────────────────────────────────────────────────────

        crop = image[y0:y1, x0:x1].copy()

        # Simple edge-based heuristic for a glasses bridge
        if crop.size:
            edges = cv2.Canny(cv2.GaussianBlur(crop, (3, 3), 0), 100, 200)
            if 255 in edges[:, edges.shape[1] // 2]:
                glasses_present = True

        # ── DEBUG overlay: draw the ROI rectangle on the full image ─────────
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imshow("Debug – ROI on full image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # ────────────────────────────────────────────────────────────────────

    return (glasses_present, crop) if return_crop else (glasses_present, None)

if __name__ == "__main__":
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:

        IMAGE_PATH = "tests/test_image.jpg"  # ← change to your file
        has_glasses, roi = detect_glasses(
            IMAGE_PATH,
            face_mesh,
            return_crop=True,
        )
        print("Glasses detected:", has_glasses)

        if roi is not None and roi.size:
            cv2.imshow("Cropped ROI", roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()