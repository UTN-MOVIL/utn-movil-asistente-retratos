# inference/core.py
from dataclasses import dataclass
from pathlib import Path
import asyncio
from typing import Callable, Iterable, Optional, Tuple, List, Any
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision

JsonMap = dict
Pts3D   = List[List[Tuple[float, float, float]]]
ImgShape = Tuple[int, int, int]

@dataclass(frozen=True)
class InferencePlugin:
    name: str
    # model provisioning
    ensure_model: Callable[[Path, Iterable[str], int], None]
    model_urls: Iterable[str]
    env_model_path_key: str            # e.g. "POSE_LANDMARKER_PATH"
    default_model_filename: str        # e.g. "pose_landmarker.task"

    # building the MP configs + landmarkers
    make_config: Callable[[Path, mp_vision.RunningMode], Any]
    factory_cls: Any                   # your LandmarkerFactory class

    # result handling
    to_json: Callable[[Any, ImgShape], JsonMap]
    points_from_result: Callable[[Any, ImgShape], Tuple[int, int, Pts3D]]
    draw_on_bgr: Optional[Callable[[np.ndarray, Any], None]] = None

    # behavior flags
    use_video_env_key: Optional[str] = None  # e.g. "POSE_USE_VIDEO"

class InferenceRuntime:
    """Holds initialized landmarkers + a lock per plugin."""
    def __init__(self, plugin: InferencePlugin, model_dir: Path):
        self.plugin = plugin
        self.model_dir = model_dir
        self.lock = asyncio.Lock()
        self.image_lmk = None
        self.video_lmk = None
        self.model_path: Optional[Path] = None

    def close(self):
        for x in (self.image_lmk, self.video_lmk):
            try:
                if x and hasattr(x, "close"):
                    x.close()
            except Exception:
                pass
        self.image_lmk = None
        self.video_lmk = None
