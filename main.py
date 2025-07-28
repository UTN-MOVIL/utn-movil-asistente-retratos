#!/usr/bin/env python3
"""
Descarga el dataset *completo* de Kaggle
    sehriyarmemmedli/glasses-vs-noglasses-dataset
(lo descomprime automáticamente) y luego evalúa la accuracy
de tu detector de gafas sobre las imágenes de train/.

Estructura esperada
└── proyecto/
    ├── kaggle.json              ← tu credencial
    ├── deteccion_lentes_v1.py    ← tu modelo
    └── download_and_eval.py      ← este archivo

Requisitos
----------
pip install kaggle tqdm numpy
"""

import os
import glob
import pathlib
import numpy as np
from tqdm import tqdm

# 1️⃣  Configurar la ruta a kaggle.json
os.environ["KAGGLE_CONFIG_DIR"] = str(pathlib.Path(__file__).parent)

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402
from deteccion_lentes_v1 import get_glasses_probability_batch  # noqa: E402

# ── Parámetros del script ────────────────────────────────────────────────────
DATASET = "sehriyarmemmedli/glasses-vs-noglasses-dataset"
DESTDIR = "data"
UMBRAL  = 0.5     # ≥ UMBRAL ⇒ “con gafas”
BATCH   = 64      # tamaño de batch para inferencia


def descargar_train(dest: str = DESTDIR) -> str:
    """Autentica, descarga y descomprime todo el dataset. Devuelve data/train/."""
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

    train_dir = os.path.join(dest, "val")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError("No se encontró la carpeta 'train/' tras descomprimir.")
    return train_dir


def evaluar(train_root: str) -> float:
    """Evalúa accuracy usando tu modelo de probabilidad de gafas."""
    with_glasses    = sorted(glob.glob(os.path.join(train_root, "with_glasses", "*")))
    without_glasses = sorted(glob.glob(os.path.join(train_root, "without_glasses", "*")))

    rutas   = with_glasses + without_glasses
    y_true  = np.array([1] * len(with_glasses) + [0] * len(without_glasses))

    print(f"[INFO] Inferencia sobre {len(rutas)} imágenes …")
    probs: list[float] = []
    for i in tqdm(range(0, len(rutas), BATCH), unit="batch"):
        probs.extend(
            get_glasses_probability_batch(rutas[i : i + BATCH], umbral_min=0.0)
        )

    y_pred = (np.array(probs) >= UMBRAL).astype(int)
    acc = (y_pred == y_true).mean()

    print(
        f"[RESULT] Accuracy: {acc*100:.2f}%  "
        f"({y_pred.sum()} con gafas / {(1 - y_pred).sum()} sin gafas)"
    )
    return acc


if __name__ == "__main__":
    train_path = descargar_train()
    evaluar(train_path)
