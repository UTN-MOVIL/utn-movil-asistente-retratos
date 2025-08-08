# medir_inclinacion_iris.py
import cv2
import math
import numpy as np
from pathlib import Path
import sys

# --- Rutas / módulos locales ---
ROOT = Path(__file__).resolve().parents[2]   # .../funcionalidades_validador_retratos
sys.path.insert(0, str(ROOT))                # añade la raíz al PYTHONPATH

from modulos.puntos_faciales import AppConfig, FaceMeshApp, DrawOptions

# ──────────────────────────────────────────────────────────────────────────────
# Face Mesh indices (lados anatómicos del sujeto)
NOSE_TIP = 1          # alt: 4
CHIN = 152
EYE_OUTER_R = 33
EYE_OUTER_L = 263
MOUTH_CORNER_R = 61
MOUTH_CORNER_L = 291

# Índices de iris (MediaPipe) – necesitan refine_landmarks=True
LEFT_IRIS_IDX  = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDX = [473, 474, 475, 476, 477]


def yaw_from_facemesh(face_landmarks, img_h, img_w,
                      camera_matrix=None, dist_coeffs=None):
    """
    Calcula yaw/pitch/roll (grados) vía PnP usando puntos anatómicos.
    Convención: yaw > 0 = el sujeto gira hacia SU izquierda.
    """
    # --- 2D pixel points desde landmarks ---
    def px(idx):
        lm = face_landmarks.landmark[idx]
        return [lm.x * img_w, lm.y * img_h]

    image_points = np.array([
        px(NOSE_TIP),          # punta de nariz
        px(CHIN),              # mentón
        px(EYE_OUTER_R),       # ojo derecho - canto externo
        px(EYE_OUTER_L),       # ojo izquierdo - canto externo
        px(MOUTH_CORNER_R),    # comisura derecha
        px(MOUTH_CORNER_L),    # comisura izquierda
    ], dtype=np.float64)

    # --- Modelo 3D simple (mm). La escala no afecta la rotación. ---
    model_points = np.array([
        [   0.0,    0.0,    0.0],   # nariz
        [   0.0, -330.0,  -65.0],   # mentón
        [ 165.0,  170.0, -135.0],   # ojo derecho externo
        [-165.0,  170.0, -135.0],   # ojo izquierdo externo
        [ 150.0, -150.0, -125.0],   # boca derecha
        [-150.0, -150.0, -125.0],   # boca izquierda
    ], dtype=np.float64)

    # --- Intrínsecos de cámara (aprox si no hay calibración) ---
    if camera_matrix is None:
        f = 1.2 * max(img_w, img_h)
        camera_matrix = np.array([[f, 0, img_w / 2],
                                  [0, f, img_h / 2],
                                  [0, 0, 1]], dtype=np.float64)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(model_points, image_points,
                                  camera_matrix, dist_coeffs,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return float('nan'), float('nan'), float('nan')

    R, _ = cv2.Rodrigues(rvec)

    # --- Matriz -> Euler (pitch-x, yaw-y, roll-z) ---
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        yaw   = math.degrees(math.atan2(-R[2, 0], sy))
        roll  = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        yaw   = math.degrees(math.atan2(-R[2, 0], sy))
        roll  = 0.0

    return yaw, pitch, roll


# ───────────────────────────────── Helpers de dibujo/geom ─────────────────────
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


# ─────────────────────────────────── Callback principal ────────────────────────
def make_medir_inclinacion_iris(flip_display=True):
    def _cb(frame, results):
        if not results or not results.multi_face_landmarks:
            return

        h, w = frame.shape[:2]
        fl = results.multi_face_landmarks[0]  # max_faces=1

        # --- Centros de iris ---
        p_left  = centro_iris(fl, LEFT_IRIS_IDX,  w, h)
        p_right = centro_iris(fl, RIGHT_IRIS_IDX, w, h)

        # marcas de iris
        cv2.circle(frame, p_left,  2, (0, 255,   0), -1)
        cv2.circle(frame, p_right, 2, (0,   0, 255), -1)

        # --- Pendiente (roll aprox por línea de iris) ---
        dx = p_right[0] - p_left[0]
        dy = p_right[1] - p_left[1]
        UMBRAL = 0.039  # sobre pendiente

        if abs(dx) >= 1e-6:
            m = dy / dx
            m_pantalla = -m if flip_display else m
            if m_pantalla <= -UMBRAL:
                roll_hint = "girar horario"
            elif m_pantalla >= UMBRAL:
                roll_hint = "girar antihorario"
            else:
                roll_hint = "sin giro en roll"
            # (opcional) imprime la sugerencia por consola
            # print(roll_hint)

        # Dibuja la recta (extendida) que une los centros de iris
        linea_infinita_por_puntos(frame, p_left, p_right, (0, 255, 255), 2)

        # ---------- Landmark fijo: 152 (mentón) ----------
        chin_idx = 152
        lx = int(fl.landmark[chin_idx].x * w)
        ly = int(fl.landmark[chin_idx].y * h)
        lx = max(0, min(w - 1, lx))
        ly = max(0, min(h - 1, ly))
        chin_pt = (lx, ly)
        cv2.circle(frame, chin_pt, 3, (0, 0, 0), -1)

        # --- Yaw/Pitch/Roll via PnP ---
        yaw_deg, pitch_deg, roll_deg = yaw_from_facemesh(fl, h, w)

        # Si la vista está espejada en pantalla, invierte el signo de yaw para UI
        yaw_ui = -yaw_deg if flip_display else yaw_deg

        if not math.isnan(yaw_ui):
            # Consola
            print(f"Yaw: {yaw_ui:.2f}°, Pitch: {pitch_deg:.2f}°, Roll: {roll_deg:.2f}°")

            # Overlay en pantalla
            cv2.putText(frame, f"Yaw: {yaw_ui:.1f} deg",  (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Pitch: {pitch_deg:.1f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Roll: {roll_deg:.1f}",  (10, 74),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)

    return _cb


# ───────────────────────────────────────── Main ───────────────────────────────
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
