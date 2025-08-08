import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define the window name
WINDOW_NAME = 'MediaPipe Face Mesh'

def dibujar_linea(image, punto1, punto2, color=(0, 255, 255), grosor=2, tipo_línea=cv2.LINE_AA):
    """
    Dibuja una línea en la imagen entre punto1 y punto2.

    Parámetros:
    - image: array de la imagen (BGR).
    - punto1: tupla (x1, y1) de coordenadas del primer punto.
    - punto2: tupla (x2, y2) de coordenadas del segundo punto.
    - color: tupla BGR para el color de la línea (por defecto amarillo).
    - grosor: grosor en píxeles de la línea.
    - tipo_línea: tipo de línea de OpenCV (por defecto cv2.LINE_AA para antialias).
    """
    cv2.line(image, punto1, punto2, color, grosor, tipo_línea)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        image.flags.writeable = True

        h, w, _ = image.shape

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Ahora solo los índices 10 y 13
            indices = [10, 13]
            colors = [
                (0, 255, 0),  # verde para 10
                (0, 0, 255)   # rojo para 13
            ]

            # Dibuja círculos y etiquetas, y guarda coordenadas
            coords = {}
            for idx, color in zip(indices, colors):
                lm = face_landmarks.landmark[idx]
                x_px, y_px = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (x_px, y_px), 5, color, -1)
                cv2.putText(image, str(idx), (x_px + 6, y_px - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                coords[idx] = (x_px, y_px)

            # Línea entre el punto 10 y el 13
            if 10 in coords and 13 in coords:
                dibujar_linea(image, coords[10], coords[13], color=(255, 0, 255), grosor=2)

        # Línea vertical central de pantalla
        puntoA = (w // 2, 0)
        puntoB = (w // 2, h)
        dibujar_linea(image, puntoA, puntoB, color=(0, 255, 255), grosor=2)

        # Muestra el frame (flip horizontal para espejo)
        cv2.imshow(WINDOW_NAME, cv2.flip(image, 1))

        # Salir con 'q' o cerrando la ventana
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()
