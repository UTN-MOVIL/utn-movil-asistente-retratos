# modulos/puntos_faciales.py
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Any

import cv2
import mediapipe as mp
import numpy as np

# Alias global reutilizable del módulo de FaceMesh
FM = mp.solutions.face_mesh


# ───────────────────────── Configuración ─────────────────────────
@dataclass
class DrawOptions:
    tesselation: bool = True
    contours: bool = True
    irises: bool = True


@dataclass
class AppConfig:
    camera_index: int = 0                # 0 = webcam por defecto
    window_name: str = "MediaPipe Face Mesh"
    max_faces: int = 1
    refine_landmarks: bool = True        # Activa iris/labios detallados
    min_det_conf: float = 0.5
    min_track_conf: float = 0.5
    flip_display: bool = True            # Espejo en pantalla
    draw: DrawOptions = field(default_factory=DrawOptions)
    input_path: Optional[str] = None     # Si se establece, usa archivo (imagen/video)
    write_video: Optional[str] = None    # Ruta para guardar video de salida (opcional)
    out_fps: int = 30
    fourcc: str = "mp4v"                 # mp4v, XVID, etc.


# ───────────────────────── Utilidades de dibujo ─────────────────────────
class Drawer:
    def __init__(self) -> None:
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

    def draw_face(self, image: np.ndarray, face_landmarks: Any, opts: DrawOptions) -> None:
        if opts.tesselation:
            self._mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=FM.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._mp_styles.get_default_face_mesh_tesselation_style(),
            )

        if opts.contours:
            self._mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=FM.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._mp_styles.get_default_face_mesh_contours_style(),
            )

        if opts.irises:
            self._mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=FM.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._mp_styles.get_default_face_mesh_iris_connections_style(),
            )


# ───────────────────────── Núcleo de la app ─────────────────────────
class FaceMeshApp:
    """
    Punto de entrada modular. Puedes pasar un callback `on_after_draw`
    para ejecutar lógica adicional por frame (mediciones, overlays, logs, etc).
    """
    def __init__(
        self,
        config: AppConfig,
        on_after_draw: Optional[Callable[[np.ndarray, Any], None]] = None,
    ) -> None:
        self.cfg = config
        self.drawer = Drawer()
        self.on_after_draw = on_after_draw

        self._cap: Optional[cv2.VideoCapture] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._last_t = time.time()
        self._fps = 0.0

    # ── Inicialización de captura (webcam o archivo) ──
    def _init_capture(self) -> None:
        if self.cfg.input_path:
            self._cap = cv2.VideoCapture(self.cfg.input_path)
        else:
            self._cap = cv2.VideoCapture(self.cfg.camera_index)

        if not self._cap or not self._cap.isOpened():
            raise RuntimeError("No se pudo abrir la fuente de video.")

        if self.cfg.write_video:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*self.cfg.fourcc)
            self._writer = cv2.VideoWriter(self.cfg.write_video, fourcc, self.cfg.out_fps, (w, h))

    # ── Actualiza y pinta el FPS ──
    def _draw_fps(self, image: np.ndarray) -> None:
        now = time.time()
        dt = now - self._last_t
        self._last_t = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps > 0 else (1.0 / dt)

        cv2.putText(
            image, f"FPS: {self._fps:0.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
        )

    # ── Bucle principal ──
    def run(self) -> None:
        self._init_capture()

        with FM.FaceMesh(
            max_num_faces=self.cfg.max_faces,
            refine_landmarks=self.cfg.refine_landmarks,
            min_detection_confidence=self.cfg.min_det_conf,
            min_tracking_confidence=self.cfg.min_track_conf,
        ) as face_mesh:

            while self._cap.isOpened():
                ok, frame = self._cap.read()
                if not ok:
                    # Para una imagen estática, salimos tras el primer frame.
                    break

                # Procesamiento
                frame.flags.writeable = False
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                # Dibujo
                frame.flags.writeable = True
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        self.drawer.draw_face(frame, face_landmarks, self.cfg.draw)

                # Hook opcional para lógica extra (mediciones/overlays/logs)
                if self.on_after_draw:
                    try:
                        self.on_after_draw(frame, results)
                    except Exception as ex:
                        # No tumbar el loop por un error en el callback
                        print(f"[WARN] on_after_draw lanzó excepción: {ex}")

                # FPS y salida
                self._draw_fps(frame)

                out = cv2.flip(frame, 1) if self.cfg.flip_display else frame
                cv2.imshow(self.cfg.window_name, out)

                if self._writer:
                    # Escribe lo mismo que ves (respeta flip si aplica)
                    self._writer.write(out)

                key = cv2.waitKey(5) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(self.cfg.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

        self.release()

    # ── Liberación de recursos ──
    def release(self) -> None:
        if self._cap:
            self._cap.release()
        if self._writer:
            self._writer.release()
        cv2.destroyAllWindows()


# ───────────────────────── Funciones de medición reutilizables ─────────────────────────
def chin_to_top_distance_px_from_landmarks(face_landmarks, w: int, h: int) -> float:
    """
    Distancia VERTICAL (px) entre el mentón (LM 152) y el punto más alto visible
    (mínimo y entre landmarks). Usa coordenadas ya normalizadas multiplicadas por w/h.
    """
    CHIN_IDX = 152
    lms = face_landmarks.landmark
    y_chin = lms[CHIN_IDX].y * h
    top_idx = min(range(len(lms)), key=lambda i: lms[i].y)
    y_top = lms[top_idx].y * h
    return float(abs(y_chin - y_top))


def porcentaje_rostro_desde_landmarks(
    face_landmarks,
    w: int,
    h: int,
    usar_convhull: bool = False
) -> float:
    """
    Calcula el % de rostro (área rostro / área imagen * 100) directamente desde landmarks.
    - usar_convhull=False -> usa bounding box de los landmarks (más estable/rápido).
    - usar_convhull=True  -> usa el convex hull de los landmarks (más fiel al contorno).
    """
    lms = face_landmarks.landmark
    xs = np.array([lm.x * w for lm in lms], dtype=np.float32)
    ys = np.array([lm.y * h for lm in lms], dtype=np.float32)

    if usar_convhull:
        pts = np.stack([xs, ys], axis=1).astype(np.float32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts)
        area_rostro = float(cv2.contourArea(hull))
    else:
        x_min = max(0, int(np.floor(xs.min())))
        x_max = min(w - 1, int(np.ceil(xs.max())))
        y_min = max(0, int(np.floor(ys.min())))
        y_max = min(h - 1, int(np.ceil(ys.max())))
        area_rostro = float(max(0, x_max - x_min) * max(0, y_max - y_min))

    area_total = float(w * h)
    return (area_rostro / area_total) * 100.0 if area_total > 0 else 0.0


def FUN_CALCULAR_PORCENTAJE_ROSTRO(
    pImagen: np.ndarray,
    C_intAreaFoto: int | None = None,
    usar_convhull: bool = False,
    min_detection_confidence: float = 0.5
) -> float:
    """
    Alternativa de una sola imagen: procesa FaceMesh internamente.
    Devuelve % de rostro respecto al área de la imagen.
    """
    if pImagen is None or pImagen.size == 0:
        raise ValueError("pImagen está vacía o es None.")

    h, w = pImagen.shape[:2]
    area_total = C_intAreaFoto if C_intAreaFoto is not None else (w * h)
    if area_total <= 0:
        raise ValueError("El área total de la foto debe ser > 0.")

    imagen_rgb = cv2.cvtColor(pImagen, cv2.COLOR_BGR2RGB)

    with FM.FaceMesh(
        static_image_mode=True,
        refine_landmarks=False,
        max_num_faces=1,
        min_detection_confidence=min_detection_confidence
    ) as face_mesh:
        results = face_mesh.process(imagen_rgb)

    if not results.multi_face_landmarks:
        return 0.0

    return porcentaje_rostro_desde_landmarks(results.multi_face_landmarks[0], w, h, usar_convhull)


# ───────────────────────── Ejemplo de overlay ─────────────────────────
def ejemplo_overlay(frame: np.ndarray, results: Any) -> None:
    """
    Ejemplo de callback: muestra cuántas caras detectadas en la esquina.
    Aquí puedes calcular ángulos, distancias, validaciones, etc.
    """
    n = len(results.multi_face_landmarks) if results and results.multi_face_landmarks else 0
    msg = f"Caras: {n}"
    cv2.putText(frame, msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)


if __name__ == "__main__":
    cfg = AppConfig(
        camera_index=0,
        window_name="MediaPipe Face Mesh",
        max_faces=1,
        refine_landmarks=True,
        min_det_conf=0.5,
        min_track_conf=0.5,
        flip_display=True,
        # input_path="ruta/a/tu/video_o_imagen",
        # write_video="salida.mp4", out_fps=30, fourcc="mp4v",
    )

    app = FaceMeshApp(cfg, on_after_draw=ejemplo_overlay)
    app.run()
