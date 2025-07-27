#!/usr/bin/env python3
"""
Script: deteccion_lentes_autodistill.py
Descripción: Detección de lentes en imágenes usando Autodistill + Detic.

Requisitos:
    pip install autodistill-detic supervision opencv-python torch torchvision tqdm
    (y clonar el repo oficial de Detic en una sub-carpeta ./Detic)
"""
import os
import sys
import urllib.request
import pathlib
import tqdm
import torch
import supervision as sv
import cv2                                 # opcional
from pathlib import Path
import shutil                              # ← nuevo
from autodistill_detic import DETIC
from autodistill.detection import CaptionOntology

# ───────────────── 1. Checkpoint ──────────────────────────────────────────────
WEIGHTS_NAME = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
WEIGHTS_URL  = f"https://dl.fbaipublicfiles.com/detic/{WEIGHTS_NAME}"

CACHE_DIR     = Path.home() / ".cache" / "autodistill" / "Detic" / "models"
WEIGHTS_LOCAL = CACHE_DIR / WEIGHTS_NAME

if not WEIGHTS_LOCAL.exists():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Pesos NO encontrados → descargando {WEIGHTS_NAME} …")
    with urllib.request.urlopen(WEIGHTS_URL) as resp, open(WEIGHTS_LOCAL, "wb") as f:
        total = int(resp.headers["Content-Length"])
        with tqdm.tqdm(total=total, unit="B", unit_scale=True) as pbar:
            while (chunk := resp.read(8192)):
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"[INFO] ✅  Pesos descargados en {WEIGHTS_LOCAL}")
else:
    print(f"[INFO] ✅  Pesos ya presentes en {WEIGHTS_LOCAL}")

# ───────────────── 2. Rutas de Detic ──────────────────────────────────────────
detic_root = Path(__file__).parent / "Detic"        # ajusta si cambia
cfg_file   = detic_root / "configs" / WEIGHTS_NAME.replace(".pth", ".yaml")

print(f"[INFO] Detic root           : {detic_root}")
print(f"[INFO] Existe Detic root?   : {detic_root.is_dir()}")
print(f"[INFO] YAML de configuración: {cfg_file}")
print(f"[INFO] Existe YAML?         : {cfg_file.exists()}")

# Hacemos visible el repo Detic para los imports de Detectron2
if str(detic_root) not in sys.path:
    sys.path.insert(0, str(detic_root))
    print(f"[INFO] sys.path ← {detic_root}")

# ───────────────── 3. Copiar YAML al directorio de trabajo ───────────────────
local_cfg_dir = Path.cwd() / "configs"
local_cfg_dir.mkdir(exist_ok=True)

target_yaml = local_cfg_dir / cfg_file.name
if not target_yaml.exists():
    shutil.copy(cfg_file, target_yaml)
    print(f"[INFO] Copiado YAML → {target_yaml}")
else:
    print(f"[INFO] YAML ya presente en {target_yaml}")

# ───────────────── 4. Construir modelo Detic ─────────────────────────────────
ontology     = CaptionOntology({"eyeglasses": "eyeglasses"})
print("[INFO] Creando modelo DETIC …")
detic_model  = DETIC(ontology=ontology)             # ya no revienta
device       = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Modelo inicializado; dispositivo detectado → {device}")

# ───────────────── 5. Utilidades de inferencia ───────────────────────────────
def get_glasses_probability(ruta_imagen: str, umbral_min: float = 0.0) -> float:
    """
    Devuelve la mayor probabilidad de detección de lentes en la imagen.
    """
    print(f"[INFO] Inferencia sobre {ruta_imagen}")
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"La ruta {ruta_imagen!r} no existe")

    detections: sv.Detections = detic_model.predict(ruta_imagen)
    print(f"[INFO] Detecciones brutas: {detections.confidence}")
    confs = detections.confidence or []
    confs = [c for c in confs if c >= umbral_min]
    print(f"[INFO] Confianzas filtradas (≥{umbral_min}): {confs}")
    return max(confs, default=0.0)

def verificar_presencia_de_lentes(ruta_imagen: str, umbral: float = 0.5) -> str:
    """
    Devuelve un mensaje legible indicando si la imagen contiene lentes.
    """
    prob = get_glasses_probability(ruta_imagen, umbral_min=umbral)
    msg  = (
        f"Imagen contiene lentes (prob. ≈ {prob:.2f})"
        if prob >= umbral
        else f"Imagen NO contiene lentes (prob. ≈ {prob:.2f})"
    )
    print(f"[INFO] Resultado final: {msg}")
    return msg

# # ───────────────── 6. Ejemplo de uso ─────────────────────────────────────────
# if __name__ == "__main__":
#     ruta = (
#         r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE"
#         r"\PROYECTO_FOTOGRAFIAS_ESTUDIANTES\datasets\validated_color\0104651666.jpg"
#     )
#     verificar_presencia_de_lentes(ruta)
