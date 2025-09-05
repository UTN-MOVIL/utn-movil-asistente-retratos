# inference/face_plugin.py
from pathlib import Path
from mediapipe.tasks.python import vision as mp_vision
from modules.puntos_faciales import (
    AppConfig as FaceAppConfig,
    LandmarkerFactory as FaceLandmarkerFactory,
    ensure_file as ensure_face_model,
    DEFAULT_MODEL_URLS as FACE_MODEL_URLS,
    draw_landmarks_bgr as face_draw_landmarks,
)
from .core import InferencePlugin

def _face_make_cfg(model_path: Path, mode: mp_vision.RunningMode):
    return FaceAppConfig(
        model_path=model_path, model_urls=list(FACE_MODEL_URLS),
        delegate_preference="gpu", running_mode=mode, max_faces=1,
        min_face_detection_confidence=0.7,
    )

def _face_to_json(result, img_shape):
    h, w = img_shape[:2]
    if not result or not getattr(result, "face_landmarks", None):
        return {"faces": [], "image_size": {"w": w, "h": h}, "num_faces": 0}
    faces = []
    for lms in result.face_landmarks:
        faces.append([{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z),
                       "px": float(lm.x * w), "py": float(lm.y * h)} for lm in lms])
    return {"faces": faces, "image_size": {"w": w, "h": h}, "num_faces": len(faces)}

def _face_points_from_result(result, img_shape):
    h, w = img_shape[:2]
    if not result or not getattr(result, "face_landmarks", None):
        return w, h, []
    out = []
    for lms in result.face_landmarks:
        pts = []
        for lm in lms:
            x_px = float(max(0, min(w - 1, lm.x * w)))
            y_px = float(max(0, min(h - 1, lm.y * h)))
            pts.append((x_px, y_px, float(lm.z)))
        out.append(pts)
    return w, h, out

FACE_PLUGIN = InferencePlugin(
    name="face",
    ensure_model=ensure_face_model,
    model_urls=FACE_MODEL_URLS,
    env_model_path_key="FACE_LANDMARKER_PATH",
    default_model_filename="face_landmarker.task",
    make_config=_face_make_cfg,
    factory_cls=FaceLandmarkerFactory,
    to_json=_face_to_json,
    points_from_result=_face_points_from_result,
    draw_on_bgr=face_draw_landmarks,
    use_video_env_key=None,  # reuse IMAGE for video path, if you want
)
