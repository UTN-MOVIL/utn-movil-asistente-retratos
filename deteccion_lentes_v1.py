#!/usr/bin/env python3
"""
Detección de lentes en imágenes usando Autodistill + Detic (OPTIMIZADO).

Requisitos:
    pip install autodistill-detic supervision opencv-python torch torchvision tqdm
    (y clonar el repo oficial de Detic en una sub-carpeta ./Detic)
"""
from __future__ import annotations

import os
import sys
import urllib.request
import shutil
from pathlib import Path
from typing import Iterable, Dict, List

import cv2
import numpy as np
import torch
import tqdm
import supervision as sv
from autodistill_detic import DETIC
from autodistill.detection import CaptionOntology


# ─────────────────────────── 1. Pesos ──────────────────────────────────────────
WEIGHTS_NAME = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
WEIGHTS_URL = f"https://dl.fbaipublicfiles.com/detic/{WEIGHTS_NAME}"
CACHE_DIR = Path.home() / ".cache" / "autodistill" / "Detic" / "models"
WEIGHTS_LOCAL = CACHE_DIR / WEIGHTS_NAME

if not WEIGHTS_LOCAL.exists():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Descargando {WEIGHTS_NAME} …")
    with urllib.request.urlopen(WEIGHTS_URL) as r, open(WEIGHTS_LOCAL, "wb") as f:
        total = int(r.headers["Content-Length"])
        with tqdm.tqdm(total=total, unit="B", unit_scale=True) as pbar:
            while chunk := r.read(8192):
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"[INFO] ✅  Pesos en {WEIGHTS_LOCAL}")
else:
    print(f"[INFO] ✅  Pesos ya presentes en {WEIGHTS_LOCAL}")

# ───────────────────── 2. Rutas / YAML de Detic ───────────────────────────────
detic_root = Path(__file__).parent / "Detic"
cfg_file = detic_root / "configs" / WEIGHTS_NAME.replace(".pth", ".yaml")
print(f"[INFO] Detic root    : {detic_root} | existe → {detic_root.is_dir()}")
print(f"[INFO] YAML original : {cfg_file}  | existe → {cfg_file.exists()}")

if str(detic_root) not in sys.path:
    sys.path.insert(0, str(detic_root))
    print(f"[INFO] sys.path ← {detic_root}")

local_cfg_dir = Path.cwd() / "configs"
local_cfg_dir.mkdir(exist_ok=True)
target_yaml = local_cfg_dir / cfg_file.name
if not target_yaml.exists():
    shutil.copy(cfg_file, target_yaml)
    print(f"[INFO] Copiado YAML → {target_yaml}")
else:
    print(f"[INFO] YAML ya presente en {target_yaml}")

print(f"[INFO] Directorio de trabajo actual → {Path.cwd()}")

# ────────────────── 3. Construcción del modelo Detic ──────────────────────────
ontology = CaptionOntology({"eyeglasses": "eyeglasses"})
print("[INFO] Creando modelo DETIC …")
detic_model = DETIC(ontology=ontology)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Modelo listo en {device}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print("[INFO] Optimizaciones CUDA activadas")

# ───────────────────────────── CACHÉS ─────────────────────────────────────────
_image_cache: Dict[str, np.ndarray] = {}
_preprocessed_cache: Dict[str, np.ndarray] = {}
_result_cache: Dict[str, float] = {}
MAX_CACHE_SIZE = 100


# ─────────────────────── utilidades internas ─────────────────────────────────
def _get_image_hash(ruta_imagen: str) -> str:
    try:
        st = os.stat(ruta_imagen)
        return f"{ruta_imagen}_{st.st_mtime}_{st.st_size}"
    except FileNotFoundError:
        return ruta_imagen


def _load_image_optimized(ruta_imagen: str) -> np.ndarray:
    img_hash = _get_image_hash(ruta_imagen)
    # limpia parte del caché cuando crece demasiado
    if len(_image_cache) > MAX_CACHE_SIZE:
        for key in list(_image_cache)[: MAX_CACHE_SIZE // 5]:
            _image_cache.pop(key)

    if img_hash not in _image_cache:
        if not os.path.exists(ruta_imagen):
            raise FileNotFoundError(f"Imagen no encontrada: {ruta_imagen}")
        img = cv2.imread(ruta_imagen)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")

        h, w = img.shape[:2]
        target = 640
        if max(h, w) > target:
            scale = target / max(h, w)
            img = cv2.resize(
                img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
            )
            print(f"[INFO] Redimensionada {ruta_imagen} a ≤{target}px")

        _image_cache[img_hash] = img

    return _image_cache[img_hash]


# ──────────────────────────── API (uno-por-uno) ──────────────────────────────
@torch.inference_mode()
def get_glasses_probability(ruta_imagen: str, umbral_min: float = 0.0) -> float:
    img_hash = _get_image_hash(ruta_imagen)
    cache_key = f"{img_hash}_{umbral_min}"
    if cache_key in _result_cache:
        return _result_cache[cache_key]

    # carga + preprocesamiento
    if img_hash in _preprocessed_cache:
        img = _preprocessed_cache[img_hash]
    else:
        img = _load_image_optimized(ruta_imagen)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # seguro que ya es RGB
        _preprocessed_cache[img_hash] = img
        if len(_preprocessed_cache) > MAX_CACHE_SIZE:
            for k in list(_preprocessed_cache)[: MAX_CACHE_SIZE // 5]:
                _preprocessed_cache.pop(k)

    # inferencia
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        dets: sv.Detections = detic_model.predict(img)

    confs = np.asarray(dets.confidence)
    if confs.size == 0:
        result = 0.0
    else:
        valid = confs[confs >= umbral_min]
        result = float(valid.max()) if valid.size else 0.0

    _result_cache[cache_key] = result
    if len(_result_cache) > MAX_CACHE_SIZE * 2:
        for k in list(_result_cache)[: MAX_CACHE_SIZE // 3]:
            _result_cache.pop(k)

    return result


# ───────────────────────────── API (batch) ────────────────────────────────────
def get_glasses_probability_batch(
    rutas_imagenes: Iterable[str], umbral_min: float = 0.0, batch_size: int = 8
) -> List[float]:
    rutas = list(rutas_imagenes)
    resultados: List[float] = [0.0] * len(rutas)

    # Load all images first with better error handling
    batch_imgs, idxs = [], []
    for i, ruta in enumerate(rutas):
        try:
            img = _load_image_optimized(ruta)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch_imgs.append(img)
            idxs.append(i)
        except Exception as e:
            print(f"[ERROR] No se pudo cargar {ruta}: {e}")

    if not batch_imgs:
        return resultados

    with torch.inference_mode(), torch.cuda.amp.autocast(
        enabled=torch.cuda.is_available()
    ):
        # Process in batches instead of one by one
        for batch_start in range(0, len(batch_imgs), batch_size):
            batch_end = min(batch_start + batch_size, len(batch_imgs))
            current_batch = batch_imgs[batch_start:batch_end]
            current_idxs = idxs[batch_start:batch_end]
            
            # Check if your model supports batch prediction
            if hasattr(detic_model, 'predict_batch'):
                # Use batch prediction if available
                batch_dets = detic_model.predict_batch(current_batch)
                for j, dets in enumerate(batch_dets):
                    confs = np.asarray(dets.confidence)
                    if confs.size:
                        valid = confs[confs >= umbral_min]
                        resultados[current_idxs[j]] = float(valid.max()) if valid.size else 0.0
            else:
                # Fallback to individual processing but in smaller batches
                for j, img in enumerate(current_batch):
                    dets: sv.Detections = detic_model.predict(img)
                    confs = np.asarray(dets.confidence)
                    if confs.size:
                        # Vectorized operations
                        valid_mask = confs >= umbral_min
                        if valid_mask.any():
                            resultados[current_idxs[j]] = float(confs[valid_mask].max())

    return resultados

# ──────────────────────── Helpers de alto nivel ──────────────────────────────
def verificar_presencia_de_lentes(ruta_imagen: str, umbral: float = 0.45) -> str:
    prob = get_glasses_probability(ruta_imagen, umbral)
    msg = (
        f"Imagen contiene lentes (prob.≈{prob:.2f})"
        if prob >= umbral
        else f"Imagen NO contiene lentes (prob.≈{prob:.2f})"
    )
    print(f"[INFO] {msg}")
    return msg


def procesar_lote_imagenes(
    rutas_imagenes: Iterable[str],
    umbral: float = 0.45,
    mostrar_progreso: bool = True,
    usar_batch: bool = True,
) -> Dict[str, float]:
    rutas = list(rutas_imagenes)

    # ——— batch ———
    if usar_batch:
        probs = get_glasses_probability_batch(rutas, umbral)
        resultados = dict(zip(rutas, probs))
        if mostrar_progreso:
            for ruta, p in resultados.items():
                estado = "✓ LENTES" if p >= umbral else "✗ sin lentes"
                print(f"[INFO] {ruta}: {estado} ({p:.2f})")
        return resultados

    # ——— uno-por-uno ———
    resultados: Dict[str, float] = {}

    if mostrar_progreso:
        bar = tqdm.tqdm(rutas, desc="Imágenes")
        for ruta in bar:
            try:
                prob = get_glasses_probability(ruta, umbral)
                resultados[ruta] = prob
                estado = "✓ LENTES" if prob >= umbral else "✗ sin lentes"
                bar.set_postfix_str(f"{estado} ({prob:.2f})")
            except Exception as e:
                print(f"[ERROR] {ruta}: {e}")
                resultados[ruta] = 0.0
    else:
        for ruta in rutas:
            try:
                resultados[ruta] = get_glasses_probability(ruta, umbral)
            except Exception as e:
                print(f"[ERROR] {ruta}: {e}")
                resultados[ruta] = 0.0

    return resultados


# ────────────────────────── utilidades varias ────────────────────────────────
def limpiar_cache_imagenes():
    _image_cache.clear()
    _preprocessed_cache.clear()
    _result_cache.clear()
    print("[INFO] Todos los cachés limpiados")


def obtener_estadisticas_cache():
    import sys

    raw_mem = sum(sys.getsizeof(img) for img in _image_cache.values()) / 1024**2
    prep_mem = (
        sum(sys.getsizeof(img) for img in _preprocessed_cache.values()) / 1024**2
    )
    print("[INFO] Estadísticas de caché:")
    print(f"  - Imágenes raw          : {len(_image_cache)}")
    print(f"  - Imágenes preprocesadas: {len(_preprocessed_cache)}")
    print(f"  - Resultados            : {len(_result_cache)}")
    print(f"  - Memoria estimada      : {raw_mem + prep_mem:.1f} MB")


def configurar_optimizaciones_gpu():
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        detic_model.predict(dummy)
        print("[INFO] GPU lista:", torch.cuda.get_device_name(0))
    else:
        print("[INFO] GPU no disponible – usando CPU")


def warm_up_modelo(iters: int = 3):
    print(f"[INFO] Pre-calentando modelo ({iters} iteraciones)…")
    sizes = [(480, 640), (320, 480), (600, 800)]
    for i in range(iters):
        h, w = sizes[i % len(sizes)]
        dummy = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        detic_model.predict(dummy)
    print("[INFO] Modelo pre-calentado")


# ────────────────────────────── CLI simple ───────────────────────────────────
if __name__ == "__main__":
    configurar_optimizaciones_gpu()
    warm_up_modelo()

    print(
        "\n[INFO] Script listo. Funciones disponibles:\n"
        "  • get_glasses_probability(ruta, umbral_min)\n"
        "  • get_glasses_probability_batch(rutas, umbral_min)\n"
        "  • verificar_presencia_de_lentes(ruta, umbral)\n"
        "  • procesar_lote_imagenes(rutas, umbral, usar_batch=True)\n"
        "  • obtener_estadisticas_cache()\n"
        "  • limpiar_cache_imagenes()\n"
        "  • configurar_optimizaciones_gpu()\n"
        "  • warm_up_modelo()"
    )

    obtener_estadisticas_cache()