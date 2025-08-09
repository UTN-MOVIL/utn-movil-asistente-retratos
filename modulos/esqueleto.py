# modulos/esqueleto.py
import cv2
import time
from dataclasses import dataclass
from typing import Optional

import mediapipe as mp

# ───────────────────────── Config ─────────────────────────
@dataclass
class PoseConfig:
    window_name: str = "Pose skeleton"
    camera_index: int = 0
    flip_display: bool = False            # True → efecto espejo
    model_complexity: int = 1             # 0=Lite, 1=Full, 2=Heavy
    min_det_conf: float = 0.5
    min_track_conf: float = 0.5
    draw_landmarks: bool = True
    show_fps: bool = True
    width: Optional[int] = None           # por ej. 1280
    height: Optional[int] = None          # por ej. 720

# ───────────────────────── Core (inferencia + dibujo) ─────────────────────────
class PoseTracker:
    def __init__(self, cfg: PoseConfig):
        self.cfg = cfg
        self._mp_pose = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._pose = None

    def __enter__(self):
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.cfg.model_complexity,
            enable_segmentation=False,
            min_detection_confidence=self.cfg.min_det_conf,
            min_tracking_confidence=self.cfg.min_track_conf,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._pose:
            self._pose.close()
        # cv2.destroyAllWindows() se hace afuera en el runner

    def process(self, bgr_frame):
        """Devuelve results de MediaPipe para un frame BGR."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        return self._pose.process(rgb)

    def draw(self, frame, results):
        """Dibuja el esqueleto sobre frame (BGR)."""
        if not results.pose_landmarks or not self.cfg.draw_landmarks:
            return
        self._mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self._mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self._mp_styles.get_default_pose_landmarks_style(),
        )

# ───────────────────────── Utilidades de UI ─────────────────────────
def put_fps(frame, fps: float, org=(10, 30)):
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

def window_is_closed(win_name: str) -> bool:
    try:
        return cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True

# ───────────────────────── Runners ─────────────────────────
def run_webcam(cfg: PoseConfig):
    cap = cv2.VideoCapture(cfg.camera_index)
    if cfg.width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    if cfg.height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)

    cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)

    prev = time.time()
    fps = 0.0

    with PoseTracker(cfg) as tracker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            if cfg.flip_display:
                frame = cv2.flip(frame, 1)

            results = tracker.process(frame)
            tracker.draw(frame, results)

            # FPS simple
            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                # amortiguar un poco
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            if cfg.show_fps:
                put_fps(frame, fps)

            cv2.imshow(cfg.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if window_is_closed(cfg.window_name):
                break

    cap.release()
    cv2.destroyAllWindows()

def run_on_video(input_path: str, cfg: PoseConfig, output_path: Optional[str] = None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)
    prev = time.time()
    fps_vis = 0.0

    with PoseTracker(cfg) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if cfg.flip_display:
                frame = cv2.flip(frame, 1)

            results = tracker.process(frame)
            tracker.draw(frame, results)

            # FPS visual
            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                fps_vis = 0.9 * fps_vis + 0.1 * (1.0 / dt)
            if cfg.show_fps:
                put_fps(frame, fps_vis)

            # Inicializa writer con el tamaño del primer frame
            if output_path and writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, 30, (w, h), True)

            if writer:
                writer.write(frame)

            cv2.imshow(cfg.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or window_is_closed(cfg.window_name):
                break

    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

def run_on_image(image_path: str, cfg: PoseConfig, save_path: Optional[str] = None):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"No se pudo leer la imagen: {image_path}")

    if cfg.flip_display:
        img = cv2.flip(img, 1)

    with PoseTracker(cfg) as tracker:
        results = tracker.process(img)
        tracker.draw(img, results)

    if cfg.show_fps:
        put_fps(img, 0.0)

    cv2.imshow(cfg.window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, img)

# ───────────────────────── Main (ejemplos) ─────────────────────────
if __name__ == "__main__":
    # 1) Webcam
    cfg = PoseConfig(
        window_name="Pose skeleton",
        camera_index=0,
        flip_display=True,
        model_complexity=1,
        min_det_conf=0.5,
        min_track_conf=0.5,
        draw_landmarks=True,
        show_fps=True,
        # width=1280, height=720,
    )
    run_webcam(cfg)

    # 2) Video file (descomenta para usar)
    # run_on_video(r"C:\ruta\a\video.mp4", cfg, output_path=r"C:\ruta\salida.mp4")

    # 3) Imagen estática (descomenta para usar)
    # run_on_image(r"C:\ruta\a\imagen.jpg", cfg, save_path=r"C:\ruta\salida.jpg")
