import cv2
import math
import mediapipe as mp

# Initialize MediaPipe Face Mesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Índices de iris (requiere refine_landmarks=True) ---
LEFT_IRIS_IDX = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDX = [473, 474, 475, 476, 477]

# Define the window name
WINDOW_NAME = 'MediaPipe Face Mesh (single image)'

# >>>>>> 1) PATH de la imagen a procesar (usa r"" para Windows) <<<<<<
PATH = r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE\TESIS\CODIGO\funcionalidades_validador_retratos\tests\test_image.jpg"

def dibujar_linea(image, punto1, punto2, color=(0, 255, 255), grosor=2, tipo_línea=cv2.LINE_AA):
    cv2.line(image, punto1, punto2, color, grosor, tipo_línea)

def centro_iris(face_landmarks, idxs, w, h):
    xs = [face_landmarks.landmark[i].x * w for i in idxs]
    ys = [face_landmarks.landmark[i].y * h for i in idxs]
    return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

def linea_infinita_por_puntos(image, p1, p2, color=(0, 255, 255), grosor=2):
    """Dibuja la línea que pasa por p1 y p2, extendida a todo el frame."""
    h, w = image.shape[:2]
    x1, y1 = p1
    x2, y2 = p2

    if abs(x2 - x1) < 1e-6:
        # Línea casi vertical
        x = max(0, min(w - 1, x1))
        cv2.line(image, (x, 0), (x, h - 1), color, grosor, cv2.LINE_AA)
        return

    m = (y2 - y1) / (x2 - x1)         # pendiente
    y_left  = int(round(y1 + m * (0 - x1)))
    y_right = int(round(y1 + m * ((w - 1) - x1)))

    # Clip suave (OpenCV recorta igual)
    y_left  = max(-h*2, min(h*3, y_left))
    y_right = max(-h*2, min(h*3, y_right))

    cv2.line(image, (0, y_left), (w - 1, y_right), color, grosor, cv2.LINE_AA)

# ---------- Procesamiento de UNA imagen ----------
# Lee la imagen
image_bgr = cv2.imread(PATH)
if image_bgr is None:
    raise FileNotFoundError(f"No se pudo leer la imagen en: {PATH}")

# Convierte a RGB para MediaPipe
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # habilita iris y labios
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    results = face_mesh.process(image_rgb)

# Trabaja sobre una copia para dibujar
image = image_bgr.copy()

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # (Opcional) Dibuja malla/contornos/iris
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        h, w = image.shape[:2]
        p_left  = centro_iris(face_landmarks, LEFT_IRIS_IDX, w, h)
        p_right = centro_iris(face_landmarks, RIGHT_IRIS_IDX, w, h)

        # Marcas de los centros (opcional)
        cv2.circle(image, p_left, 2, (0, 255, 0), -1)    # verde
        cv2.circle(image, p_right, 2, (0, 0, 255), -1)   # rojo

        # ---- Pendiente de la línea que cruza ambos iris ----
        dx = p_right[0] - p_left[0]
        dy = p_right[1] - p_left[1]

        if abs(dx) < 1e-6:
            pendiente = math.inf  # línea prácticamente vertical
        else:
            pendiente = dy / dx

        # Como NO hay flip en la visualización, imprimimos la pendiente tal cual
        if math.isfinite(pendiente):
            print(f"Pendiente: {pendiente:.4f}")
        else:
            print("Pendiente: ∞ (línea vertical)")

        # Dibuja la línea
        linea_infinita_por_puntos(image, p_left, p_right, color=(0, 255, 255), grosor=2)
else:
    print("No se detectó ningún rostro en la imagen.")

# Muestra y/o guarda el resultado
cv2.imshow(WINDOW_NAME, image)  # sin flip
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite("salida_iris_linea.jpg", image)  # opcional
