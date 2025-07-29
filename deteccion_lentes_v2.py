#!/usr/bin/env python3
# glasses_detector_cnn.py

import numpy as np
import dlib
import cv2
from pathlib import Path
from PIL import Image

# --- Carga de modelos ---
# Resuelve la ruta base donde se encuentra el script
BASE_DIR = Path(__file__).resolve().parent

# --- Carga del detector facial CNN ---
# Este modelo es más preciso que el HoG y puede usar GPU.
# Debes descargar el archivo 'mmod_human_face_detector.dat' y colocarlo en la carpeta 'models'.
MODEL_CNN_PATH = BASE_DIR / "models" / "mmod_human_face_detector.dat"
detector = dlib.cnn_face_detection_model_v1(str(MODEL_CNN_PATH))

# --- Carga del predictor de puntos faciales (landmark predictor) ---
# Este modelo sigue siendo el mismo.
MODEL_LANDMARKS_PATH = BASE_DIR / "models" / "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(str(MODEL_LANDMARKS_PATH))

def glasses_detector(path: str):
    """
    Detecta si en la imagen indicada hay gafas usando un detector facial CNN.
    La lógica se basa en la presencia de un puente nasal en los bordes
    centrados del recorte de la zona nasal.

    Devuelve:
      - 'No face detected' si no encuentra cara
      - 1 si detecta gafas
      - 0 si no detecta gafas
    """
    # Carga la imagen en formato RGB compatible con dlib
    img = dlib.load_rgb_image(path)

    # Detecta caras usando el modelo CNN. El '1' indica que se debe
    # sobremuestrear la imagen 1 vez para detectar caras más pequeñas.
    faces = detector(img, 1)
    if len(faces) == 0:
        return 'No face detected'

    # Tomamos la primera cara detectada.
    # El detector CNN devuelve un objeto 'mmod_rectangle', por lo que accedemos
    # al rectángulo de la cara con el atributo .rect
    rect = faces[0].rect
    
    sp = predictor(img, rect)
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])

    # Índices del puente nasal (puntos 28–35)
    nose_idxs = [28, 29, 30, 31, 33, 34, 35]
    nose_bridge_x = [landmarks[i][0] for i in nose_idxs]
    nose_bridge_y = [landmarks[i][1] for i in nose_idxs]

    # Coordenadas del recorte de la zona nasal
    x_min, x_max = min(nose_bridge_x), max(nose_bridge_x)
    y_min = landmarks[20][1]  # altura aproximada de la ceja
    y_max = landmarks[30][1]  # final del puente

    # Recorta la zona nasal
    pil_img = Image.open(path)
    # Se añade un pequeño try-except por si las coordenadas son inválidas
    try:
        cropped = pil_img.crop((x_min, y_min, x_max, y_max))
    except ValueError:
        # Esto puede ocurrir si las coordenadas del recorte no son válidas
        return 0 

    # Suaviza y detecta bordes
    img_blur = cv2.GaussianBlur(np.array(cropped), (3, 3), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

    # Si la matriz de bordes está vacía, no hay nada que analizar
    if edges.size == 0:
        return 0
        
    # Columna central de bordes
    center_col = edges.T[edges.shape[1] // 2]

    # Si hay un píxel blanco (255) en la columna central, interpretamos que hay un puente => gafas
    return 1 if 255 in center_col else 0