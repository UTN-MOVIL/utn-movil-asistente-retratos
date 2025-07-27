#!/usr/bin/env python3
"""
Detección de lentes en imágenes usando Autodistill + Detic (OPTIMIZADO).
Requisitos:
    pip install autodistill-detic supervision opencv-python torch torchvision tqdm
    (y clonar el repo oficial de Detic en una sub-carpeta ./Detic)
"""
import os
import sys
import urllib.request
import shutil
from pathlib import Path
from typing import Iterable, Dict, List
import torch
import tqdm
import numpy as np
import supervision as sv
import cv2
from autodistill_detic import DETIC
from autodistill.detection import CaptionOntology

# ───────────────── 1. Pesos ──────────────────────────────────────────────────
WEIGHTS_NAME  = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
WEIGHTS_URL   = f"https://dl.fbaipublicfiles.com/detic/{WEIGHTS_NAME}"
CACHE_DIR     = Path.home()/".cache"/"autodistill"/"Detic"/"models"
WEIGHTS_LOCAL = CACHE_DIR/WEIGHTS_NAME

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

# ───────────────── 2. Rutas Detic ────────────────────────────────────────────
detic_root = Path(__file__).parent/"Detic"
cfg_file   = detic_root/"configs"/WEIGHTS_NAME.replace(".pth", ".yaml")
print(f"[INFO] Detic root    : {detic_root} | existe → {detic_root.is_dir()}")
print(f"[INFO] YAML original : {cfg_file}  | existe → {cfg_file.exists()}")

if str(detic_root) not in sys.path:
    sys.path.insert(0, str(detic_root))
    print(f"[INFO] sys.path ← {detic_root}")

local_cfg_dir = Path.cwd()/"configs"
local_cfg_dir.mkdir(exist_ok=True)
target_yaml = local_cfg_dir/cfg_file.name
if not target_yaml.exists():
    shutil.copy(cfg_file, target_yaml)
    print(f"[INFO] Copiado YAML → {target_yaml}")
else:
    print(f"[INFO] YAML ya presente en {target_yaml}")

print(f"[INFO] Directorio de trabajo actual → {Path.cwd()}")

# ─────────── 3. Construir modelo Detic ───────────────────────────────────────
ontology    = CaptionOntology({"eyeglasses": "eyeglasses"})
print("[INFO] Creando modelo DETIC …")
detic_model = DETIC(ontology=ontology)
device      = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Modelo listo en {device}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False
    print("[INFO] Optimizaciones CUDA activadas")

_image_cache        = {}
_preprocessed_cache = {}
_result_cache       = {}
MAX_CACHE_SIZE      = 100

def _get_image_hash(ruta_imagen: str) -> str:
    try:
        stat = os.stat(ruta_imagen)
        return f"{ruta_imagen}_{stat.st_mtime}_{stat.st_size}"
    except:
        return ruta_imagen

def _load_image_optimized(ruta_imagen: str) -> np.ndarray:
    global _image_cache
    img_hash = _get_image_hash(ruta_imagen)
    if len(_image_cache) > MAX_CACHE_SIZE:
        for key in list(_image_cache)[:MAX_CACHE_SIZE//5]:
            del _image_cache[key]
    if img_hash not in _image_cache:
        if not os.path.exists(ruta_imagen):
            raise FileNotFoundError(f"Imagen no encontrada: {ruta_imagen}")
        img = cv2.imread(ruta_imagen)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
        h, w = img.shape[:2]
        target_size = 640
        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            new_w, new_h = int(w*scale), int(h*scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            print(f"[INFO] Redimensionada {ruta_imagen}: {w}x{h} → {new_w}x{new_h}")
        _image_cache[img_hash] = img
    return _image_cache[img_hash]

@torch.inference_mode()
def get_glasses_probability(ruta_imagen: str, umbral_min: float = 0.0) -> float:
    global _result_cache, _preprocessed_cache
    try:
        img_hash = _get_image_hash(ruta_imagen)
        cache_key = f"{img_hash}_{umbral_min}"
        if cache_key in _result_cache:
            return _result_cache[cache_key]

        # Carga/preproceso con caché
        if img_hash in _preprocessed_cache:
            img = _preprocessed_cache[img_hash]
        else:
            img = _load_image_optimized(ruta_imagen)
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _preprocessed_cache[img_hash] = img
            if len(_preprocessed_cache) > MAX_CACHE_SIZE:
                for key in list(_preprocessed_cache)[:MAX_CACHE_SIZE//5]:
                    del _preprocessed_cache[key]

        # Inferencia
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            dets: sv.Detections = detic_model.predict(img)

        # Procesamiento de confianzas
        if not dets.confidence:
            result = 0.0
        else:
            confs = np.array(dets.confidence)
            valid_confs = confs[confs >= umbral_min]
            result = float(np.max(valid_confs)) if valid_confs.size else 0.0

        # Guardar en caché
        _result_cache[cache_key] = result
        if len(_result_cache) > MAX_CACHE_SIZE * 2:
            for key in list(_result_cache)[:MAX_CACHE_SIZE//3]:
                del _result_cache[key]

        return result

    except Exception as e:
        print(f"[ERROR] Error procesando {ruta_imagen}: {e}")
        return 0.0

def get_glasses_probability_batch(rutas_imagenes: Iterable[str], umbral_min: float = 0.0) -> List[float]:
    resultados = [0.0] * len(list(rutas_imagenes))
    try:
        batch_imgs = []
        idxs = []
        for i, ruta in enumerate(rutas_imagenes):
            try:
                img = _load_image_optimized(ruta)
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_imgs.append(img)
                idxs.append(i)
            except Exception as e:
                print(f"[ERROR] No se pudo cargar {ruta}: {e}")

        if not batch_imgs:
            return resultados

        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            for j, img in enumerate(batch_imgs):
                dets = detic_model.predict(img)
                if dets.confidence:
                    confs = np.array(dets.confidence)
                    valid = confs[confs >= umbral_min]
                    resultados[idxs[j]] = float(np.max(valid)) if valid.size else 0.0

    except Exception as e:
        print(f"[ERROR] Error en procesamiento batch: {e}")

    return resultados

def verificar_presencia_de_lentes(ruta_imagen: str, umbral: float = 0.5) -> str:
    prob = get_glasses_probability(ruta_imagen, umbral)
    msg = (
        f"Imagen contiene lentes (prob.≈{prob:.2f})"
        if prob >= umbral else
        f"Imagen NO contiene lentes (prob.≈{prob:.2f})"
    )
    print(f"[INFO] {msg}")
    return msg

def procesar_lote_imagenes(
    rutas_imagenes: Iterable[str],
    umbral: float = 0.5,
    mostrar_progreso: bool = True,
    usar_batch: bool = True
) -> Dict[str, float]:
    resultados: Dict[str, float] = {}

    if usar_batch:
        probs = get_glasses_probability_batch(rutas_imagenes, umbral)
        resultados = dict(zip(rutas_imagenes, probs))
        if mostrar_progreso:
            for ruta, prob in resultados.items():
                estado = "✓ LENTES" if prob >= umbral else "✗ sin lentes"
                print(f"[INFO] {ruta}: {estado} ({prob:.2f})")

    else:
        if mostrar_progreso:
            bar = tqdm.tqdm(rutas_imagenes, desc="Procesando imágenes")
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
            for ruta in rutas_imagenes:
                try:
                    resultados[ruta] = get_glasses_probability(ruta, umbral)
                except Exception as e:
                    print(f"[ERROR] {ruta}: {e}")
                    resultados[ruta] = 0.0

    return resultados

def limpiar_cache_imagenes():
    global _image_cache, _preprocessed_cache, _result_cache
    _image_cache.clear()
    _preprocessed_cache.clear()
    _result_cache.clear()
    print("[INFO] Todos los cachés limpiados")

def obtener_estadisticas_cache():
    print("[INFO] Estadísticas de caché:")
    print(f"  - Imágenes raw: {len(_image_cache)}")
    print(f"  - Imágenes preprocesadas: {len(_preprocessed_cache)}")
    print(f"  - Resultados: {_result_cache and len(_result_cache)}")
    import sys
    mem_raw = sum(sys.getsizeof(img) for img in _image_cache.values()) / 1024**2
    mem_prep = sum(sys.getsizeof(img) for img in _preprocessed_cache.values()) / 1024**2
    print(f"[INFO] Memoria estimada: {mem_raw + mem_prep:.1f} MB")

def configurar_optimizaciones_gpu():
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            dummy = np.zeros((480,640,3), dtype=np.uint8)
            _ = detic_model.predict(dummy)
            print("[INFO] GPU pre-calentada con inferencia dummy")
        except:
            print("[INFO] No se pudo pre-calententar GPU")
        print("[INFO] Optimizaciones GPU adicionales activadas")
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Memoria GPU libre: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    else:
        print("[INFO] GPU no disponible - usando CPU")

def warm_up_modelo(num_iteraciones: int = 3):
    print(f"[INFO] Pre-calentando modelo con {num_iteraciones} iteraciones...")
    tamaños = [(480,640), (320,480), (600,800)]
    for i in range(num_iteraciones):
        h, w = tamaños[i % len(tamaños)]
        dummy = np.random.randint(0,255,(h,w,3),dtype=np.uint8)
        with torch.inference_mode():
            _ = detic_model.predict(dummy)
    print("[INFO] Modelo pre-calentado")

if __name__ == "__main__":
    configurar_optimizaciones_gpu()
    warm_up_modelo()
    print("[INFO] Script ultra-optimizado listo. Funciones disponibles:")
    print("  - get_glasses_probability(ruta, umbral_min)")
    print("  - get_glasses_probability_batch(lista_rutas, umbral_min)")
    print("  - verificar_presencia_de_lentes(ruta, umbral)")
    print("  - procesar_lote_imagenes(lista_rutas, umbral, usar_batch=True)")
    print("  - obtener_estadisticas_cache()")
    print("  - limpiar_cache_imagenes()")
    print("  - configurar_optimizaciones_gpu()")
    print("  - warm_up_modelo()")
    obtener_estadisticas_cache()