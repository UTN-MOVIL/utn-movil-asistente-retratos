#!/usr/bin/env python3
# glasses_detector_gpu.py (Refactorizado para GPU y mejor modularidad)

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Dict, List, Union, Optional

import cv2
import dlib
import numpy as np
import tqdm

# ─────────────────── 1. Carga de modelos y configuración global ───────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SHAPE_PREDICTOR_PATH = MODELS_DIR / "shape_predictor_68_face_landmarks.dat"
CNN_FACE_DETECTOR_PATH = MODELS_DIR / "mmod_human_face_detector.dat"

# --- Variables Globales para los modelos y configuración ---
detector: Optional[callable] = None
predictor: Optional[dlib.shape_predictor] = None
GPU_ENABLED = False

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
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _image_cache[img_hash] = img_rgb
    
    return _image_cache[img_hash]

def _get_landmarks_for_image(img_rgb: np.ndarray) -> Optional[np.ndarray]:
    """Función interna para detectar cara y obtener landmarks de una sola imagen."""
    if detector is None or predictor is None:
        raise RuntimeError("Los modelos no han sido cargados. Llama a 'configurar_optimizaciones_gpu' primero.")

    # El detector CNN devuelve objetos mmod_rectangle, el HOG devuelve rectángulos simples
    if GPU_ENABLED:
        faces = detector(img_rgb, 1) # `1` = upsample para mayor precisión
        if faces:
            # Extraer el rectángulo del objeto mmod_rectangle
            face_rect = faces[0].rect
            shape = predictor(img_rgb, face_rect)
            return np.array([[p.x, p.y] for p in shape.parts()])
    else:
        faces = detector(img_rgb)
        if faces:
            shape = predictor(img_rgb, faces[0])
            return np.array([[p.x, p.y] for p in shape.parts()])
    
    return None

def _calculate_prob_from_landmarks(img_rgb: np.ndarray, landmarks: np.ndarray) -> float:
    """Calcula la probabilidad de lentes a partir de los landmarks y la imagen."""
    nose_bridge_points = landmarks[27:31]
    x_coords, y_coords = nose_bridge_points[:, 0], nose_bridge_points[:, 1]
    
    padding = 5
    x_min, x_max = np.min(x_coords) - padding, np.max(x_coords) + padding
    y_min, y_max = landmarks[27][1] - padding, landmarks[30][1] + padding

    # Recortar usando escala de grises para la detección de bordes
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cropped_nose_area = img_gray[y_min:y_max, x_min:x_max]

    if cropped_nose_area.size == 0:
        return 0.0

    img_blur = cv2.GaussianBlur(cropped_nose_area, (5, 5), 0)
    edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=150)

    center_x = edges.shape[1] // 2
    center_cols = edges[:, max(0, center_x - 1):center_x + 2]

    return np.count_nonzero(center_cols) / center_cols.size if center_cols.size > 0 else 0.0

# ─────────────────── 4. Funciones de gestión y configuración ──────────────────
def configurar_optimizaciones_gpu(use_gpu: bool = True):
    """
    Configura y carga los modelos de dlib para usar GPU (CNN) o CPU (HOG).
    Esta función debe ser llamada antes que cualquier otra función de procesamiento.
    """
    global detector, predictor, GPU_ENABLED
    
    if not SHAPE_PREDICTOR_PATH.exists():
        print(f"[ERROR] Modelo de landmarks no encontrado en: {SHAPE_PREDICTOR_PATH}")
        sys.exit(1)

    print("[INFO] Cargando modelo de predicción de landmarks...")
    predictor = dlib.shape_predictor(str(SHAPE_PREDICTOR_PATH))
    
    if use_gpu:
        if not dlib.DLIB_USE_CUDA:
            print("[WARN] dlib no fue compilado con soporte para CUDA. Cambiando a modo CPU.")
            configurar_optimizaciones_gpu(use_gpu=False)
            return

        if not CNN_FACE_DETECTOR_PATH.exists():
            print(f"[ERROR] Modelo de detector facial CNN no encontrado: {CNN_FACE_DETECTOR_PATH}")
            print("[INFO] Asegúrate de descargar 'mmod_human_face_detector.dat'.")
            sys.exit(1)
        
        print("[INFO] Cargando detector facial CNN para uso con GPU...")
        detector = dlib.cnn_face_detection_model_v1(str(CNN_FACE_DETECTOR_PATH))
        GPU_ENABLED = True
        print("[INFO] ✅ Detección por GPU (CUDA) habilitada.")
    else:
        print("[INFO] Cargando detector facial HOG para uso con CPU...")
        detector = dlib.get_frontal_face_detector()
        GPU_ENABLED = False
        print("[INFO] ✅ Detección por CPU habilitada.")

def warm_up_modelo():
    """
    Ejecuta una inferencia en vacío para cargar los modelos en la memoria de la GPU
    y evitar la latencia inicial en el primer procesamiento real.
    """
    if detector is None:
        print("[ERROR] Los modelos no están cargados. Llama a 'configurar_optimizaciones_gpu' primero.")
        return
        
    print("[INFO] Calentando el modelo (warm-up)...")
    # Crear una imagen dummy para la primera pasada
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        # La detección en sí misma fuerza la carga del modelo en la VRAM
        detector(dummy_image, 1 if GPU_ENABLED else 0)
        print("[INFO] ✅ Modelo listo para inferencia rápida.")
    except Exception as e:
        print(f"[ERROR] Falló el calentamiento del modelo: {e}")

# ────────────────────── 5. API Principal ───────────────────────────
def get_glasses_probability(path: str) -> float: # <--- Change type hint to float
    """
    Estima la probabilidad de que haya gafas en una única imagen.
    Optimizado con múltiples capas de caché.
    """
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
            img_rgb = _load_image_optimized(path)
            landmarks = _get_landmarks_for_image(img_rgb)
            _landmarks_cache[img_hash] = landmarks

        if landmarks is None:
            _result_cache[img_hash] = 0.0
            return 0.0  # <--- FIX 1: Was 'No face detected'

        # Para el cálculo, se necesita la imagen original (podría no estar en caché de imagen)
        img_rgb_for_calc = _load_image_optimized(path)
        glasses_prob = _calculate_prob_from_landmarks(img_rgb_for_calc, landmarks)
        
        _result_cache[img_hash] = glasses_prob
        return glasses_prob

    except (FileNotFoundError, IOError) as e:
        print(f"[WARN] Error de archivo procesando {path}: {e}")
        return 0.0 # <--- FIX 2: Was str(e)
    except Exception as e:
        print(f"[ERROR] Error inesperado procesando {path}: {e}")
        return 0.0 # <--- FIX 3: Was "Processing error"

def get_glasses_probability_batch(
    rutas_imagenes: List[str],
    batch_size: int = 32, # Un batch_size de 16, 32 o 64 es un buen punto de partida
) -> Dict[str, float]:
    """
    Procesa un lote de imágenes de forma optimizada y con uso eficiente de memoria,
    cargando y procesando las imágenes en lotes pequeños.
    """
    if detector is None:
        raise RuntimeError("Los modelos no han sido cargados. Llama a 'configurar_optimizaciones_gpu' primero.")

    resultados: Dict[str, float] = {}
    
    # 1. Cargar resultados del caché primero (sin cambios)
    rutas_a_procesar_inicial = []
    for ruta in rutas_imagenes:
        img_hash = _get_image_hash(ruta)
        if img_hash in _result_cache:
            resultados[ruta] = _result_cache[img_hash]
        else:
            rutas_a_procesar_inicial.append(ruta)

    if not rutas_a_procesar_inicial:
        return resultados

    print(f"[INFO] Procesando {len(rutas_a_procesar_inicial)} imágenes nuevas en {len(rutas_a_procesar_inicial) // batch_size + 1} lotes...")

    # 2. Iterar sobre las rutas en lotes pequeños
    for i in tqdm.tqdm(range(0, len(rutas_a_procesar_inicial), batch_size), desc="Procesando Lotes"):
        # Obtiene las rutas para el lote actual
        batch_paths = rutas_a_procesar_inicial[i : i + batch_size]
        
        # Carga solo las imágenes de ESTE lote
        batch_images = []
        valid_batch_paths = []
        for ruta in batch_paths:
            try:
                img = _load_image_optimized(ruta)
                batch_images.append(img)
                valid_batch_paths.append(ruta)
            except IOError:
                tqdm.tqdm.write(f"\n⚠️ [ADVERTENCIA] Se omitirá el archivo corrupto: {ruta}")
                resultados[ruta] = -2.0
                _result_cache[_get_image_hash(ruta)] = -2.0
        
        if not valid_batch_paths:
            continue # Si todas las imágenes del lote estaban corruptas, pasar al siguiente

        # 3. Procesa el lote actual (con la memoria ya controlada)
        if GPU_ENABLED:
            # Detección por lotes en la GPU
            all_faces = detector(batch_images, 1, batch_size=len(batch_images)) # El batch_size aquí es el del lote actual
            
            # Extraer landmarks para el lote actual
            for ruta, img_rgb, faces in zip(valid_batch_paths, batch_images, all_faces):
                landmarks = None
                if faces:
                    shape = predictor(img_rgb, faces[0].rect)
                    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
                _landmarks_cache[_get_image_hash(ruta)] = landmarks
        else:
            # Procesamiento en CPU (uno por uno)
            for ruta, img_rgb in zip(valid_batch_paths, batch_images):
                landmarks = _get_landmarks_for_image(img_rgb)
                _landmarks_cache[_get_image_hash(ruta)] = landmarks

    # 4. Calcular probabilidades para todas las imágenes procesadas (sin cambios)
    rutas_finales = [r for r in rutas_a_procesar_inicial if r not in resultados] # Rutas que no fallaron al cargar
    for ruta in rutas_finales:
        landmarks = _landmarks_cache.get(_get_image_hash(ruta))
        if landmarks is not None:
            img_rgb = _load_image_optimized(ruta)
            prob = _calculate_prob_from_landmarks(img_rgb, landmarks)
            resultados[ruta] = prob
            _result_cache[_get_image_hash(ruta)] = prob
        else:
            resultados[ruta] = -1.0 # No face detected
            _result_cache[_get_image_hash(ruta)] = -1.0
            
    return resultados

def obtener_estadisticas_cache():
    """Imprime el estado actual de los cachés."""
    print("[INFO] Estadísticas de caché:")
    print(f"  - Imágenes cargadas   : {len(_image_cache)} / {MAX_CACHE_SIZE}")
    print(f"  - Puntos faciales     : {len(_landmarks_cache)} / {MAX_CACHE_SIZE}")
    print(f"  - Resultados          : {len(_result_cache)} / {MAX_CACHE_SIZE * 2}")