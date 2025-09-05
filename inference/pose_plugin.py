# inference/pose_plugin.py
from pathlib import Path
import os
from mediapipe.tasks.python import vision as mp_vision
from modules.esqueleto import (
    AppConfig as PoseAppConfig,
    LandmarkerFactory as PoseLandmarkerFactory,
    ensure_file as ensure_pose_model,
    DEFAULT_MODEL_URLS as POSE_MODEL_URLS,
    draw_pose_skeleton_bgr,
)
from .core import InferencePlugin
from typing import Tuple, List

def _pose_make_cfg(model_path: Path, mode: mp_vision.RunningMode):
    return PoseAppConfig(
        model_path=model_path,
        model_urls=list(POSE_MODEL_URLS),
        delegate_preference="gpu",
        running_mode=mode,
        max_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.2 if mode == mp_vision.RunningMode.VIDEO else None,
    )

def _pose_to_json(result, img_shape):
    h, w = img_shape[:2]
    if not result or not getattr(result, "pose_landmarks", None):
        return {"poses": [], "image_size": {"w": w, "h": h}, "num_poses": 0}
    poses = []
    for lms in result.pose_landmarks:
        poses.append([{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z),
                       "px": float(lm.x * w), "py": float(lm.y * h),
                       "visibility": float(getattr(lm, "visibility", 0.0))}
                      for lm in lms])
    return {"poses": poses, "image_size": {"w": w, "h": h}, "num_poses": len(poses)}

def _pose_points_from_result(result, img_shape):
    h, w = img_shape[:2]
    if not result or not getattr(result, "pose_landmarks", None):
        return w, h, []
    worlds = getattr(result, "pose_world_landmarks", None)
    out: List[List[Tuple[float, float, float]]] = []
    for i, lms in enumerate(result.pose_landmarks):
        pts = []
        world_i = worlds[i] if (worlds and len(worlds) > i and worlds[i]) else None
        for j, lm in enumerate(lms):
            x_px = float(max(0, min(w - 1, lm.x * w)))
            y_px = float(max(0, min(h - 1, lm.y * h)))
            z    = float(world_i[j].z) if world_i else float(lm.z)
            pts.append((x_px, y_px, z))
        out.append(pts)
    return w, h, out

POSE_PLUGIN = InferencePlugin(
    name="pose",
    ensure_model=ensure_pose_model,
    model_urls=POSE_MODEL_URLS,
    env_model_path_key="POSE_LANDMARKER_PATH",
    default_model_filename="pose_landmarker.task",
    make_config=_pose_make_cfg,
    factory_cls=PoseLandmarkerFactory,
    to_json=_pose_to_json,
    points_from_result=_pose_points_from_result,
    draw_on_bgr=draw_pose_skeleton_bgr,
    use_video_env_key="POSE_USE_VIDEO",
)
