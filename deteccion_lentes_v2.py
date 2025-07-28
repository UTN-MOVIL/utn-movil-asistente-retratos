#!/usr/bin/env python3
# glasses_detector.py (OPTIMIZADO)

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Dict, List, Union

import cv2
import dlib
import numpy as np
import tqdm
from PIL import Image

# ─────────────────── 1. Carga de modelos y rutas ────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "shape_predictor_68_face_landmarks.dat"

if not MODEL_PATH.exists():
    print(f"[ERROR] Modelo no encontrado en: {MODEL_PATH}")
    print("[INFO] Asegúrate de descargar 'shape_predictor_68_face_landmarks.dat' y colocarlo en la carpeta 'models'.")
    sys.exit(1)

print("[INFO] Cargando modelos de dlib...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(MODEL_PATH))
print("[INFO] ✅ Modelos listos.")

# ───────────────────────────── 2. CACHÉS ───────────────────────────────────
_image_cache: Dict[str, np.ndarray] = {}
_landmarks_cache: Dict[str, Union[np.ndarray, None]] = {}
_result_cache: Dict[str, float] = {}
MAX_CACHE_SIZE = 200 # Aumentado para el caché de landmarks

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
        # Elimina el 20% más antiguo de las claves
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
        
        # Cargar con OpenCV y convertir a RGB (compatible con dlib)
        img = cv2.imread(ruta_imagen)
        if img is None:
            raise IOError(f"No se pudo cargar la imagen: {ruta_imagen}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Redimensionar imágenes grandes para acelerar la detección
        h, w = img.shape[:2]
        target = 1024 # dlib es lento con imágenes muy grandes
        if max(h, w) > target:
            scale = target / max(h, w)
            img = cv2.resize(
                img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
            )
        _image_cache[img_hash] = img
        
    return _image_cache[img_hash]

# ────────────────────── 4. API Principal (Optimizada) ─────────────────────
def get_glasses_probability(path: str) -> Union[float, str]:
    """
    Estima la probabilidad de que haya gafas en la imagen.
    Esta versión está optimizada con múltiples capas de caché.
    """
    try:
        img_hash = _get_image_hash(path)

        # --- Nivel 1: Cache de Resultados ---
        if img_hash in _result_cache:
            return _result_cache[img_hash]
        
        _manage_cache(_result_cache, MAX_CACHE_SIZE * 2)

        landmarks = None
        # --- Nivel 2: Cache de Puntos Faciales (Landmarks) ---
        if img_hash in _landmarks_cache:
            landmarks = _landmarks_cache[img_hash]
        else:
            _manage_cache(_landmarks_cache, MAX_CACHE_SIZE)
            # --- Nivel 3: Cache de Imagen ---
            img = _load_image_optimized(path)
            
            faces = detector(img)
            if faces:
                landmarks = np.array([[p.x, p.y] for p in predictor(img, faces[0]).parts()])
            _landmarks_cache[img_hash] = landmarks

        if landmarks is None:
            _result_cache[img_hash] = 0.0 # Considera "no cara" como "sin gafas"
            return 'No face detected'

        # --- Análisis de bordes en la región nasal (lógica original) ---
        nose_bridge_points = landmarks[27:31]
        x_coords, y_coords = nose_bridge_points[:, 0], nose_bridge_points[:, 1]
        
        padding = 5
        x_min, x_max = min(x_coords) - padding, max(x_coords) + padding
        y_min, y_max = landmarks[27][1] - padding, landmarks[30][1] + padding

        # Recortar usando la imagen en escala de grises para la detección de bordes
        img_gray = cv2.cvtColor(_image_cache[img_hash], cv2.COLOR_RGB2GRAY)
        cropped_nose_area = img_gray[y_min:y_max, x_min:x_max]

        if cropped_nose_area.size == 0:
            return 0.0

        # Suavizado y detección de bordes
        img_blur = cv2.GaussianBlur(cropped_nose_area, (5, 5), 0)
        edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=150)

        # --- Cálculo de la probabilidad ---
        center_x = edges.shape[1] // 2
        center_cols = edges[:, max(0, center_x - 1):center_x + 1]

        if center_cols.size == 0:
            glasses_prob = 0.0
        else:
            glasses_prob = np.count_nonzero(center_cols) / center_cols.size
        
        _result_cache[img_hash] = glasses_prob
        return glasses_prob

    except (FileNotFoundError, IOError) as e:
        return str(e)
    except Exception as e:
        print(f"[ERROR] Error inesperado procesando {path}: {e}")
        return "Processing error"

# ──────────────────────── 5. Helpers de alto nivel ─────────────────────────
def verificar_presencia_de_lentes(ruta_imagen: str, umbral: float = 0.35) -> str:
    """Función de alto nivel para verificar una sola imagen e imprimir el resultado."""
    prob = get_glasses_probability(ruta_imagen)
    
    if isinstance(prob, str):
        print(f"[INFO] {ruta_imagen}: {prob}")
        return prob

    msg = (
        f"Imagen contiene lentes (prob.≈{prob:.2f})"
        if prob >= umbral
        else f"Imagen NO contiene lentes (prob.≈{prob:.2f})"
    )
    print(f"[INFO] {msg}")
    return msg

def procesar_lote_imagenes(
    rutas_imagenes: Iterable[str],
    umbral: float = 0.35,
    mostrar_progreso: bool = True,
) -> Dict[str, float]:
    """Procesa una lista de imágenes y devuelve un diccionario con sus probabilidades."""
    rutas = list(rutas_imagenes)
    resultados: Dict[str, float] = {}
    
    iterable = tqdm.tqdm(rutas, desc="Procesando Imágenes") if mostrar_progreso else rutas

    for ruta in iterable:
        prob = get_glasses_probability(ruta)
        
        # Asegurarse de que el resultado sea numérico para el diccionario
        final_prob = prob if isinstance(prob, float) else -1.0
        resultados[ruta] = final_prob

        if mostrar_progreso:
            if isinstance(prob, str):
                estado_str = f"⚠️ {prob}"
            else:
                estado = "✓ LENTES" if prob >= umbral else "✗ sin lentes"
                estado_str = f"{estado} ({prob:.2f})"
            iterable.set_postfix_str(estado_str)
            
    return resultados

# ────────────────────── 6. utilidades de gestión ───────────────────────────
def limpiar_cache_imagenes():
    """Limpia todos los cachés en memoria."""
    _image_cache.clear()
    _landmarks_cache.clear()
    _result_cache.clear()
    print("[INFO] ✅ Todos los cachés han sido limpiados.")

def obtener_estadisticas_cache():
    """Imprime el estado actual de los cachés."""
    # La estimación de memoria es compleja para objetos de numpy, así que nos centramos en el conteo.
    print("[INFO] Estadísticas de caché:")
    print(f"  - Imágenes raw cargadas : {len(_image_cache)} / {MAX_CACHE_SIZE}")
    print(f"  - Puntos faciales       : {len(_landmarks_cache)} / {MAX_CACHE_SIZE}")
    print(f"  - Resultados            : {len(_result_cache)} / {MAX_CACHE_SIZE * 2}")

# ────────────────────────── 7. CLI y Ejemplo de Uso ────────────────────────
if __name__ == "__main__":
    # Reemplaza con la ruta a tu imagen o una carpeta de imágenes
    image_path = r'C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE\TESIS\datasets\validated_color\0202032710.jpg'
    
    if not Path(image_path).exists():
        print(f"[ERROR] La ruta de ejemplo no existe: {image_path}")
        print("[INFO]  Por favor, actualiza la variable 'image_path' en la sección __main__.")
    else:
        print("\n--- 1. Probando una sola imagen ---")
        verificar_presencia_de_lentes(image_path, umbral=0.35)

        print("\n--- 2. Probando la misma imagen de nuevo (debería usar caché) ---")
        verificar_presencia_de_lentes(image_path, umbral=0.35)

        # Ejemplo de procesamiento por lotes (usando la misma imagen varias veces para demostrar el caché)
        print("\n--- 3. Probando el procesamiento por lotes ---")
        image_folder = Path(image_path).parent
        # Tomar las primeras 10 imágenes para el ejemplo
        sample_images = [str(p) for p in image_folder.glob('*.jpg')][:10]
        
        if sample_images:
            resultados_lote = procesar_lote_imagenes(sample_images)
            print("\n[INFO] Resultados del lote procesado.")
        else:
            print("[INFO] No se encontraron imágenes .jpg para el ejemplo de lote.")

        print("\n--- 4. Estadísticas del caché ---")
        obtener_estadisticas_cache()

        print("\n--- 5. Limpiando caché ---")
        limpiar_cache_imagenes()
        obtener_estadisticas_cache()