#!/usr/bin/env python3
"""
Detección de lentes en imágenes usando Autodistill + Detic (versión optimizada).

Requisitos:
    pip install autodistill-detic supervision opencv-python torch torchvision tqdm
    (y clonar el repo oficial de Detic en una sub-carpeta ./Detic)
"""
from __future__ import annotations

import os
import sys
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
import supervision as sv
import torch
import tqdm
from autodistill.detection import CaptionOntology
from autodistill_detic import DETIC

# ───────────────────────────── 1. Pesos ──────────────────────────────────────
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

# ────────────────────── 2. Rutas / YAML de Detic ────────────────────────────
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

# ───────────── 3. Construcción y optimización del modelo ────────────────────
ontology = CaptionOntology({"eyeglasses": "eyeglasses"})
print("[INFO] Creando modelo DETIC …")
detic_model = DETIC(ontology=ontology)                   # wrapper

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Modelo listo en {device}")

if device == "cuda":
    detic_model.detic_model.half()                       # FP16
    detic_model.detic_model = torch.compile(detic_model.detic_model)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print("[INFO] Optimizaciones CUDA + FP16 activadas")

# ───────────────────────────── CACHÉS ───────────────────────────────────────
_image_cache: Dict[str, np.ndarray] = {}
_preprocessed_cache: Dict[str, np.ndarray] = {}
_result_cache: Dict[str, float] = {}
MAX_CACHE_SIZE = 100

# ─────────────────────── utilidades internas ────────────────────────────────
def _get_image_hash(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{path}_{st.st_mtime}_{st.st_size}"
    except FileNotFoundError:
        return path


def _load_image_optimized(path: str) -> np.ndarray:
    img_hash = _get_image_hash(path)
    if len(_image_cache) > MAX_CACHE_SIZE:
        for k in list(_image_cache)[: MAX_CACHE_SIZE // 5]:
            _image_cache.pop(k)

    if img_hash not in _image_cache:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Imagen no encontrada: {path}")
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {path}")

        h, w = img.shape[:2]
        target = 640
        if max(h, w) > target:
            scale = target / max(h, w)
            img = cv2.resize(
                img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
            )
            print(f"[INFO] Redimensionada {path} a ≤{target}px")

        _image_cache[img_hash] = img

    return _image_cache[img_hash]

# ──────────────────────────── API (uno-por-uno) ─────────────────────────────
@torch.inference_mode()
def get_glasses_probability(path: str, umbral_min: float = 0.0) -> float:
    img_hash = _get_image_hash(path)
    cache_key = f"{img_hash}_{umbral_min}"
    if cache_key in _result_cache:
        return _result_cache[cache_key]

    if img_hash in _preprocessed_cache:
        img = _preprocessed_cache[img_hash]
    else:
        img = _load_image_optimized(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _preprocessed_cache[img_hash] = img
        if len(_preprocessed_cache) > MAX_CACHE_SIZE:
            for k in list(_preprocessed_cache)[: MAX_CACHE_SIZE // 5]:
                _preprocessed_cache.pop(k)

    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        dets: sv.Detections = detic_model.predict(img)

    confs = np.asarray(dets.confidence)
    result = float(confs[confs >= umbral_min].max()) if confs.size else 0.0

    _result_cache[cache_key] = result
    if len(_result_cache) > MAX_CACHE_SIZE * 2:
        for k in list(_result_cache)[: MAX_CACHE_SIZE // 3]:
            _result_cache.pop(k)

    return result

# ───────────────────────────── API (batch) ──────────────────────────────────
def _load_and_rgb(path: str) -> np.ndarray:
    img = _load_image_optimized(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def get_glasses_probability_batch(
    paths: Iterable[str], umbral_min: float = 0.0
) -> List[float]:
    rutas = list(paths)
    resultados = [0.0] * len(rutas)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        imgs = list(pool.map(_load_and_rgb, rutas))

    inputs = [
        {
            "image": torch.from_numpy(img)
            .permute(2, 0, 1)
            .to(device)
            .half()
            if device == "cuda"
            else torch.from_numpy(img).permute(2, 0, 1),
            "height": img.shape[0],
            "width": img.shape[1],
        }
        for img in imgs
    ]

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
        outputs = detic_model.detic_model(inputs)

    for i, out in enumerate(outputs):
        confs = out["instances"].scores.cpu().numpy()
        if confs.size:
            valid = confs[confs >= umbral_min]
            resultados[i] = float(valid.max()) if valid.size else 0.0

    return resultados

# ─────────────────────── Helpers de alto nivel ──────────────────────────────
def verificar_presencia_de_lentes(path: str, umbral: float = 0.45) -> str:
    prob = get_glasses_probability(path, umbral)
    msg = (
        f"Imagen contiene lentes (prob≈{prob:.2f})"
        if prob >= umbral
        else f"Imagen NO contiene lentes (prob≈{prob:.2f})"
    )
    print(f"[INFO] {msg}")
    return msg


def procesar_lote_imagenes(
    rutas: Iterable[str],
    umbral: float = 0.45,
    mostrar_progreso: bool = True,
    usar_batch: bool = True,
) -> Dict[str, float]:
    rutas = list(rutas)

    if usar_batch:
        probs = get_glasses_probability_batch(rutas, umbral)
        resultados = dict(zip(rutas, probs))
        if mostrar_progreso:
            for pth, p in resultados.items():
                estado = "✓ LENTES" if p >= umbral else "✗ sin lentes"
                print(f"[INFO] {pth}: {estado} ({p:.2f})")
        return resultados

    resultados: Dict[str, float] = {}

    if mostrar_progreso:
        bar = tqdm.tqdm(rutas, desc="Imágenes")
        for pth in bar:
            try:
                prob = get_glasses_probability(pth, umbral)
                resultados[pth] = prob
                estado = "✓ LENTES" if prob >= umbral else "✗ sin lentes"
                bar.set_postfix({"estado": f"{estado} ({prob:.2f})"})
            except Exception as e:
                print(f"[ERROR] {pth}: {e}")
                resultados[pth] = 0.0
    else:
        for pth in rutas:
            try:
                resultados[pth] = get_glasses_probability(pth, umbral)
            except Exception as e:
                print(f"[ERROR] {pth}: {e}")
                resultados[pth] = 0.0

    return resultados

# ───────────────────────── utilidades varias ────────────────────────────────
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
    if device == "cuda":
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

# ───────────────────────────── CLI simple ───────────────────────────────────
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