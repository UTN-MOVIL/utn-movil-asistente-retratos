#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional
import cv2
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]   # .../funcionalidades_validador_retratos
sys.path.insert(0, str(ROOT))                # añade la raíz al PYTHONPATH

# Reutiliza tus utilidades del módulo común
from modulos.puntos_faciales import (
    FM,
    chin_to_top_distance_px_from_landmarks as chin_dist,
)

def distancia_menton_top(
    image: Union[str, np.ndarray],
    face_mesh: Optional["FM.FaceMesh"] = None
) -> float:
    """
    Calcula la distancia (px) desde el mentón (idx 152) al punto más alto del rostro
    en UNA imagen.

    Parámetros:
        image: ruta del archivo (str) o frame BGR (np.ndarray).
        face_mesh: instancia opcional de FM.FaceMesh para reutilizar entre llamadas.

    Devuelve:
        float: distancia en píxeles.

    Excepciones:
        FileNotFoundError, ValueError, RuntimeError en casos de error/ausencia de rostro.
    """
    # 1) Cargar imagen si viene como ruta
    if isinstance(image, str):
        frame = cv2.imread(image)
        if frame is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {image}")
    else:
        frame = image
        if frame is None or not hasattr(frame, "shape"):
            raise ValueError("El parámetro 'image' no es un ndarray BGR válido.")

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2) Procesar con FaceMesh (reutilizable u on-demand)
    if face_mesh is None:
        with FM.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as fm:
            res = fm.process(rgb)
    else:
        res = face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        raise RuntimeError("No se detectó rostro en la imagen.")

    # 3) Calcular con tu helper reutilizado
    face_lms = res.multi_face_landmarks[0]
    dist_px = float(chin_dist(face_lms, w, h))
    return dist_px

# Ejemplo rápido:
dist = distancia_menton_top(r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE\TESIS\CODIGO\funcionalidades_validador_retratos\results\image_cache\1716222136.jpg")
print("Distancia px:", dist)
