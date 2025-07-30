#!/usr/bin/env python3
# glasses_detector.py

import numpy as np
import dlib
import cv2
from pathlib import Path
from PIL import Image

# --- Carga de modelos ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_CNN_PATH = BASE_DIR / "models" / "mmod_human_face_detector.dat"
detector = dlib.cnn_face_detection_model_v1(str(MODEL_CNN_PATH))

# Ruta portátil al modelo de 68 puntos de referencia faciales
MODEL_PATH = BASE_DIR / "models" / "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(str(MODEL_PATH))

def glasses_detector(path: str):
    """
    Detecta si en la imagen indicada hay gafas basándose en la presencia
    de un puente nasal en los bordes centrados del recorte de la zona nasal.
    Devuelve:
      - 'No face detected' si no encuentra cara
      - 1 si detecta gafas
      - 0 si no detecta gafas
    """
    # Carga la imagen en formato RGB compatible con dlib
    img = dlib.load_rgb_image(path)
    faces = detector(img)
    if len(faces) == 0:
        return 'No face detected'

    # Tomamos la primera cara detectada
    rect = faces[0]
    sp = predictor(img, rect)
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])

    # Índices del puente nasal (puntos 28–35)
    nose_idxs = [28, 29, 30, 31, 33, 34, 35]
    nose_bridge_x = [landmarks[i][0] for i in nose_idxs]
    nose_bridge_y = [landmarks[i][1] for i in nose_idxs]

    # Coordenadas del recorte de la zona nasal
    x_min, x_max = min(nose_bridge_x), max(nose_bridge_x)
    y_min = landmarks[20][1]   # altura aproximada de la ceja
    y_max = landmarks[30][1]   # final del puente

    # Recorta la zona nasal
    pil_img = Image.open(path)
    cropped = pil_img.crop((x_min, y_min, x_max, y_max))

    # Suaviza y detecta bordes
    img_blur = cv2.GaussianBlur(np.array(cropped), (3, 3), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

    # Columna central de bordes
    center_col = edges.T[len(edges.T) // 2]

    # Si hay píxel blanco (255), interpretamos puente => gafas
    return 1 if 255 in center_col else 0


if __name__ == "__main__":
    # Ejemplo de uso: reemplaza con la ruta a tu imagen
    image_path = "/kaggle/working/funcionalidades_validador_retratos/tests/test_image.jpg"
    result = glasses_detector(image_path)

    if result == 'No face detected':
        print("No se detectó ninguna cara en la imagen.")
    elif result == 1:
        print("Gafas detectadas ✔️")
    else:
        print("No se detectaron gafas ✖️")
