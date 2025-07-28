#!/usr/bin/env python3
# Nombre del archivo: deteccion_lentes_v2.py

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Dict, List, Union

import cv2
import dlib
import numpy as np
import tqdm

# ─────────────────── 1. Carga de modelos y rutas ────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "shape_predictor_68_face_landmarks.dat"

if not MODEL_PATH.exists():
    print(f"[ERROR] Modelo no encontrado en: {MODEL_PATH}")
    print("[INFO] Asegúrate de descargar 'shape_predictor_68_face_landmarks.dat' y colocarlo en la carpeta 'models' junto a este script.")
    sys.exit(1)

print("[INFO] Cargando modelos de dlib...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(MODEL_PATH))
print("[INFO] ✅ Modelos dlib listos (basados en CPU).")

# ───────────────────────────── 2. CACHÉS ───────────────────────────────────
_image_cache: Dict[str, np.ndarray] = {}
_landmarks_cache: Dict[str, Union[np.ndarray, None]] = {}
_result_cache: Dict[str, float] = {}
MAX_CACHE_SIZE = 200

# ─────────────────────── 3. utilidades internas ───────────────────────────
def _get_image_hash(ruta_imagen: str) -> str:
    """Genera una clave única para el caché basada en metadatos del archivo."""
    try:
        st = os.stat(ruta_imagen)
        return f"{ruta_imagen}_{st.st_mtime}_{st.st_size}"
    except FileNotFoundError:
        return ruta_imagen

def _manage_cache(cache: dict, max_size: int):
    """Limpia una parte del caché si excede el tamaño máximo."""
    if len(cache) > max_size:
        keys_to_remove = list(cache.keys())[:max_size // 5]
        for key in keys_to_remove:
            cache.pop(key, None)

def _load_image_optimized(ruta_imagen: str) -> np.ndarray:
    """Carga una imagen desde el caché o disco, redimensionándola si es necesario."""
    img_hash = _get_image_hash(ruta_imagen)
    _manage_cache(_image_cache, MAX_CACHE_SIZE)

    if img_hash not in _image_cache:
        if not os.path.exists(ruta_imagen):
            raise FileNotFoundError(f"Imagen no encontrada: {ruta_imagen}")
        
        img = cv2.imread(ruta_imagen)
        if img is None:
            raise IOError(f"No se pudo cargar la imagen: {ruta_imagen}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        target = 1024
        if max(h, w) > target:
            scale = target / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        _image_cache[img_hash] = img
        
    return _image_cache[img_hash]

# ────────────────────── 4. API Principal (Optimizada) ─────────────────────
def get_glasses_probability(path: str, umbral_min: float = 0.0) -> Union[float, str]:
    """Estima la probabilidad de que haya gafas en la imagen."""
    # El parámetro 'umbral_min' se mantiene por compatibilidad, pero no se usa en la lógica de dlib.
    try:
        img_hash = _get_image_hash(path)

        if img_hash in _result_cache:
            return _result_cache[img_hash]
        
        _manage_cache(_result_cache, MAX_CACHE_SIZE * 2)

        landmarks = None
        if img_hash in _landmarks_cache:
            landmarks = _landmarks_cache[img_hash]
        else:
            _manage_cache(_landmarks_cache, MAX_CACHE_SIZE)
            img = _load_image_optimized(path)
            
            faces = detector(img)
            if faces:
                landmarks = np.array([[p.x, p.y] for p in predictor(img, faces[0]).parts()])
            _landmarks_cache[img_hash] = landmarks

        if landmarks is None:
            _result_cache[img_hash] = 0.0
            return 'No face detected'

        nose_bridge_points = landmarks[27:31]
        x_coords, y_coords = nose_bridge_points[:, 0], nose_bridge_points[:, 1]
        
        padding = 5
        x_min, x_max = int(min(x_coords) - padding), int(max(x_coords) + padding)
        y_min, y_max = int(landmarks[27][1] - padding), int(landmarks[30][1] + padding)

        img_gray = cv2.cvtColor(_image_cache[img_hash], cv2.COLOR_RGB2GRAY)
        cropped_nose_area = img_gray[y_min:y_max, x_min:x_max]

        if cropped_nose_area.size == 0: return 0.0

        img_blur = cv2.GaussianBlur(cropped_nose_area, (5, 5), 0)
        edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=150)

        center_x = edges.shape[1] // 2
        center_cols = edges[:, max(0, center_x - 1):center_x + 1]

        glasses_prob = np.count_nonzero(center_cols) / center_cols.size if center_cols.size > 0 else 0.0
        
        _result_cache[img_hash] = glasses_prob
        return glasses_prob

    except (FileNotFoundError, IOError) as e:
        return str(e)
    except Exception as e:
        print(f"[ERROR] Error inesperado procesando {path}: {e}")
        return "Processing error"

# ────────────────────── 5. utilidades de gestión ───────────────────────────
def limpiar_cache_imagenes():
    _image_cache.clear()
    _landmarks_cache.clear()
    _result_cache.clear()
    print("[INFO] ✅ Todos los cachés (dlib) han sido limpiados.")

def obtener_estadisticas_cache():
    print("[INFO] Estadísticas de caché (dlib):")
    print(f"  - Imágenes raw cargadas : {len(_image_cache)} / {MAX_CACHE_SIZE}")
    print(f"  - Puntos faciales       : {len(_landmarks_cache)} / {MAX_CACHE_SIZE}")
    print(f"  - Resultados            : {len(_result_cache)} / {MAX_CACHE_SIZE * 2}")


# █▀▀ █▀█ █▀▄▀█ █▀▀ ▄▀█ ▀█▀ █ █▄ █ █ █▀█ █▄ █
# █▄▄ █γκ █ ▀ █ ██▄ █▀█  █  █ █ ▀█ █ █▄█ █ ▀█
#
# --- FUNCIONES DE COMPATIBILIDAD ---
# Estas funciones permiten que el script principal funcione sin cambios.

def configurar_optimizaciones_gpu():
    """Función vacía por compatibilidad. Dlib usa CPU por defecto."""
    print("[INFO] Detector dlib (CPU) no requiere configuración de GPU.")
    pass

def warm_up_modelo():
    """Función vacía por compatibilidad. Dlib no necesita pre-calentamiento."""
    print("[INFO] Detector dlib (CPU) no necesita pre-calentamiento.")
    pass

def get_glasses_probability_batch(
    rutas_imagenes: Iterable[str], umbral_min: float = 0.0
) -> List[float]:
    """
    Adaptador para el procesamiento por lotes. Llama a la función principal
    en un bucle y devuelve una lista de probabilidades, como se espera.
    """
    resultados = []
    # Usa tqdm aquí para mostrar el progreso de la detección
    for ruta in tqdm.tqdm(list(rutas_imagenes), desc="Detectando lentes (dlib)"):
        prob = get_glasses_probability(ruta, umbral_min)
        # Si hay un error (devuelve string), se convierte en 0.0
        resultados.append(prob if isinstance(prob, float) else 0.0)
    return resultados