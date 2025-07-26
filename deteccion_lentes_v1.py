"""
Script: deteccion_lentes_autodistill.py
Descripción:
    Detección de lentes en imágenes usando Autodistill + Detic.

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
import cv2  # Opcional: solo si vas a visualizar resultados

# ── 1. Asegurar el checkpoint en la ruta que espera Detic ─────────────────────
WEIGHTS_NAME = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
WEIGHTS_URL = f"https://dl.fbaipublicfiles.com/detic/{WEIGHTS_NAME}"

CACHE_DIR = pathlib.Path.home() / ".cache" / "autodistill" / "Detic" / "models"
WEIGHTS_LOCAL = CACHE_DIR / WEIGHTS_NAME

if not WEIGHTS_LOCAL.exists():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Descargando pesos Detic (~900 MB). Esto puede tardar…")
    with urllib.request.urlopen(WEIGHTS_URL) as resp, open(WEIGHTS_LOCAL, "wb") as f:
        total = int(resp.headers["Content-Length"])
        with tqdm.tqdm(total=total, unit="B", unit_scale=True) as pbar:
            while (chunk := resp.read(8192)):
                f.write(chunk)
                pbar.update(len(chunk))
    print("✅  Checkpoint descargado en", WEIGHTS_LOCAL)

# ── 2. Hacer visible el repo original de Detic ────────────────────────────────
detic_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "Detic"))
if detic_repo_root not in sys.path:
    sys.path.insert(0, detic_repo_root)

# ── 3. Construir el modelo Autodistill Detic ──────────────────────────────────
from autodistill_detic import DETIC
from autodistill.detection import CaptionOntology

ontology = CaptionOntology({"eyeglasses": "eyeglasses"})
detic_model = DETIC(ontology=ontology)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Modelo inicializado en {device}")

# ── 4. Funciones utilitarias de inferencia ────────────────────────────────────
def get_glasses_probability(ruta_imagen: str, umbral_min: float = 0.0) -> float:
    """
    Devuelve la mayor probabilidad de detección de lentes en la imagen.

    Args:
        ruta_imagen: Ruta al archivo de imagen.
        umbral_min: Umbral mínimo de confianza aceptado (0–1).

    Returns:
        Probabilidad en rango [0, 1]. 0.0 si no hay detecciones válidas.
    """
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"La ruta {ruta_imagen!r} no existe")

    detections: sv.Detections = detic_model.predict(ruta_imagen)
    confs = detections.confidence or []
    confs = [c for c in confs if c >= umbral_min]
    return max(confs, default=0.0)


def verificar_presencia_de_lentes(ruta_imagen: str, umbral: float = 0.5) -> str:
    """
    Devuelve un mensaje legible indicando si la imagen contiene lentes.
    """
    prob = get_glasses_probability(ruta_imagen, umbral_min=umbral)
    if prob >= umbral:
        return f"Imagen contiene lentes (prob. ≈ {prob:.2f})"
    return f"Imagen NO contiene lentes (prob. ≈ {prob:.2f})"

# # ── 5. Ejemplo de uso ----------------------------------------------------------
# if __name__ == "__main__":
#     ruta = (
#         r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE"
#         r"\PROYECTO_FOTOGRAFIAS_ESTUDIANTES\datasets\validated_color\0104651666.jpg"
#     )
#     print("Resultado final:", verificar_presencia_de_lentes(ruta))
