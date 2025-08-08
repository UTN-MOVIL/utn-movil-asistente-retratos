# medir_inclinacion_iris.py
import cv2
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]   # .../funcionalidades_validador_retratos
sys.path.insert(0, str(ROOT))                # añade la raíz al PYTHONPATH

from modulos.puntos_faciales import AppConfig, FaceMeshApp, DrawOptions

# Índices de iris (MediaPipe) – necesitan refine_landmarks=True
LEFT_IRIS_IDX  = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDX = [473, 474, 475, 476, 477]

def centro_iris(face_landmarks, idxs, w, h):
    xs = [face_landmarks.landmark[i].x * w for i in idxs]
    ys = [face_landmarks.landmark[i].y * h for i in idxs]
    return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

def linea_infinita_por_puntos(image, p1, p2, color=(0, 255, 255), grosor=2):
    h, w = image.shape[:2]
    x1, y1 = p1
    x2, y2 = p2

    if abs(x2 - x1) < 1e-6:
        x = max(0, min(w - 1, x1))
        cv2.line(image, (x, 0), (x, h - 1), color, grosor, cv2.LINE_AA)
        return

    m = (y2 - y1) / (x2 - x1)
    y_left  = int(round(y1 + m * (0 - x1)))
    y_right = int(round(y1 + m * ((w - 1) - x1)))
    cv2.line(image, (0, y_left), (w - 1, y_right), color, grosor, cv2.LINE_AA)

def make_medir_inclinacion_iris(flip_display=True):
    def _cb(frame, results):
        if not results or not results.multi_face_landmarks:
            return

        h, w = frame.shape[:2]
        fl = results.multi_face_landmarks[0]  # estamos con max_faces=1

        # --- Centros de iris ---
        p_left  = centro_iris(fl, LEFT_IRIS_IDX,  w, h)
        p_right = centro_iris(fl, RIGHT_IRIS_IDX, w, h)

        # marcas opcionales de iris
        cv2.circle(frame, p_left,  2, (0, 255,   0), -1)
        cv2.circle(frame, p_right, 2, (0,   0, 255), -1)

        # --- Pendiente + aviso de giro respecto a horizontal ---
        dx = p_right[0] - p_left[0]
        dy = p_right[1] - p_left[1]
        UMBRAL = 0.039  # umbral sobre pendiente

        if abs(dx) < 1e-6:
            # línea vertical (pendiente infinita)
            pass
        else:
            m = dy / dx
            m_pantalla = -m if flip_display else m
            if m_pantalla <= -UMBRAL:
                print("girar horario")
            elif m_pantalla >= UMBRAL:
                print("girar antihorario")
            else:
                print("sin giro en roll")

        # Dibuja la recta (extendida) que une los centros de iris
        linea_infinita_por_puntos(frame, p_left, p_right, (0, 255, 255), 2)

        # ---------- Landmark fijo: 152 (mentón) ----------
        chin_idx = 152
        lx = int(fl.landmark[chin_idx].x * w)
        ly = int(fl.landmark[chin_idx].y * h)

        # Clamp por seguridad a los bordes de la imagen
        lx = max(0, min(w - 1, lx))
        ly = max(0, min(h - 1, ly))

        chin_pt = (lx, ly)

        # Dibuja el punto (puedes cambiar color/tamaño)
        cv2.circle(frame, chin_pt, 3, (0, 0, 0), -1)

        # print(f"Landmark 152 (mentón): px={chin_pt}")

    return _cb

if __name__ == "__main__":
    cfg = AppConfig(
        camera_index=0,
        window_name="MediaPipe Face Mesh",
        max_faces=1,
        refine_landmarks=True,   # importante para iris
        min_det_conf=0.5,
        min_track_conf=0.5,
        flip_display=True,       # espejo en pantalla
        draw=DrawOptions(tesselation=True, contours=True, irises=True),
        # input_path="ruta/a/tu/video_o_imagen",  # opcional: para archivo en vez de webcam
        # write_video="salida.mp4", out_fps=30, fourcc="mp4v",  # opcional: grabar
    )

    app = FaceMeshApp(cfg, on_after_draw=make_medir_inclinacion_iris(cfg.flip_display))
    app.run()
