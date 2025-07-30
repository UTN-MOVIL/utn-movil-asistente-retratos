#!/usr/bin/env python3
"""
Descarga el dataset *completo* de Kaggle
    sehriyarmemmedli/glasses-vs-noglasses-dataset
(lo descomprime automáticamente) y luego evalúa la accuracy
de tu detector de gafas sobre las imágenes de val/.

Además, genera un Excel con:
  • Ruta de la imagen (hipervínculo)
  • Probabilidad devuelta por el modelo
  • Detección (SÍ / NO) usando el umbral 0.4486

Requisitos
----------
pip install kaggle tqdm numpy
# y tu módulo `exportacion_datos_excel` en PYTHONPATH
"""

import os
import glob
import pathlib
from datetime import datetime
import numpy as np
from tqdm import tqdm

# 1️⃣  Configurar la ruta a kaggle.json
os.environ["KAGGLE_CONFIG_DIR"] = str(pathlib.Path(__file__).parent)

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402
from deteccion_lentes_v1 import get_glasses_probability_batch  # noqa: E402
from exportacion_datos_excel import (  # noqa: E402
    format_to_hyperlinks,
    normalize_dict_lengths,
    dict_to_excel,
    get_file_count,
)

# ── Parámetros del script ────────────────────────────────────────────────────
DATASET   = "sehriyarmemmedli/glasses-vs-noglasses-dataset"
DESTDIR   = "data"
UMBRAL    = 0.4486        # ≥ UMBRAL ⇒ “con gafas”
BATCH     = 64          # tamaño de batch para inferencia
RESULTS   = "results"   # carpeta de reportes Excel

# ─────────────────────────────────────────────────────────────────────────────


def descargar_val(dest: str = DESTDIR) -> str:
    """Autentica, descarga y descomprime todo el dataset. Devuelve data/val/."""
    api = KaggleApi()
    api.authenticate()

    print("[INFO] Descargando y descomprimiendo dataset completo…")
    api.dataset_download_files(
        DATASET,
        path=dest,
        unzip=True,
        quiet=False,
        force=True,
    )

    val_dir = os.path.join(dest, "val")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError("No se encontró la carpeta 'val/' tras descomprimir.")
    return val_dir


def evaluar(val_root: str) -> float:
    """Evalúa accuracy y genera un Excel con los resultados de cada imagen."""
    with_glasses    = sorted(glob.glob(os.path.join(val_root, "with_glasses", "*")))
    without_glasses = sorted(glob.glob(os.path.join(val_root, "without_glasses", "*")))

    rutas  = with_glasses + without_glasses
    y_true = np.array([1] * len(with_glasses) + [0] * len(without_glasses))

    print(f"[INFO] Inferencia sobre {len(rutas)} imágenes …")
    probs: list[float] = []
    for i in tqdm(range(0, len(rutas), BATCH), unit="batch"):
        probs.extend(
            get_glasses_probability_batch(rutas[i : i + BATCH], umbral_min=0.0)
        )

    probs_arr = np.array(probs)
    y_pred = (probs_arr >= UMBRAL).astype(int)
    acc = (y_pred == y_true).mean()

    # ── Reporte en consola ─────────────────────────────────────────────────
    print(
        f"[RESULT] Accuracy: {acc*100:.2f}%  "
        f"({y_pred.sum()} con gafas / {(1 - y_pred).sum()} sin gafas)"
    )

    # ── Generar Excel ─────────────────────────────────────────────────────
    os.makedirs(RESULTS, exist_ok=True)

    info = {
        "Ruta":       format_to_hyperlinks(rutas),
        "Probabilidad": probs,
        "Detección":  ["SÍ" if p >= UMBRAL else "NO" for p in probs],
    }
    normalized = normalize_dict_lengths(info)

    numero_reporte = get_file_count(RESULTS) + 1
    nombre_excel = (
        f"{RESULTS}/Reporte_{numero_reporte:03d}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )
    dict_to_excel(normalized, nombre_excel)

    print(f"[INFO] 📄 Excel generado: {nombre_excel}")

    return acc


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    val_path = descargar_val()
    evaluar(val_path)