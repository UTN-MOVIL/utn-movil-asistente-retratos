import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, Union

def detect_glasses(
    image_path: str,
    *,                       # keyword-only flags after this point
    return_crop: bool = False,
    show_crop: bool   = False,
) -> Union[bool, Tuple[bool, Optional[np.ndarray]]]:
    """
    Detects whether the subject in an image is wearing glasses.
    
    Parameters
    ----------
    image_path : str
        Path to the input image.
    return_crop : bool, optional
        If True, also return the cropped nose-bridge ROI. Default is False.
    show_crop : bool, optional
        If True, display debug windows with the full image + ROI and the crop.
        Requires an X/GUI environment. Default is False.
    
    Returns
    -------
    bool
        True if glasses are detected, False otherwise (when return_crop=False).
    (bool, np.ndarray | None)
        When return_crop=True, returns a tuple (has_glasses, crop).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = image.shape[:2]

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        rgb      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results  = face_mesh.process(rgb)

    glasses_present = False
    crop: Optional[np.ndarray] = None

    if results.multi_face_landmarks:
        lm  = results.multi_face_landmarks[0].landmark
        lmp = np.array([(int(p.x * w), int(p.y * h)) for p in lm])

        pts = lmp[[168, 6, 197, 195, 5, 4]]
        x, y, ww, hh = cv2.boundingRect(pts)
        pad, frac = 5, 0.5
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + ww + pad), min(h, y + int(hh * frac) + pad)

        crop = image[y0:y1, x0:x1].copy()

        if crop.size:
            edges = cv2.Canny(cv2.GaussianBlur(crop, (3, 3), 0), 100, 200)
            if 255 in edges[:, edges.shape[1] // 2]:
                glasses_present = True

        if show_crop:
            dbg = image.copy()
            cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.imshow("Debug – ROI on full image", dbg)
            if crop.size:
                cv2.imshow("Cropped ROI", crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if return_crop:
        return glasses_present, crop
    return glasses_present

has_glasses = detect_glasses("tests/test_image.jpg")
print(has_glasses)