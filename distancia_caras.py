#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from typing import List, Tuple, Any, Union
from tqdm import tqdm

import cv2
import mediapipe as mp
import numpy as np

# 🔁 Solo importamos lo que sí usaremos (validación local de imágenes)
from modulos.preprocesamiento import (
    process_image_list,   # ← mantiene tu validación/borrado de corruptos si quieres
)

from modulos.exportacion_datos_excel import (
    format_to_hyperlinks,
    normalize_dict_lengths,
    dict_to_excel,
    get_file_count,
)

CACHE_DIR = "image_cache"   # ya no se usa para descargar, pero lo dejamos por compatibilidad

# ----------------------- Cálculo con MediaPipe -----------------------

CHIN_IDX = 152  # Índice de mentón (MediaPipe Face Mesh)

def _chin_to_top_distance_px_from_landmarks(face_landmarks, w: int, h: int) -> float:
    """
    Devuelve la distancia VERTICAL (px) entre el mentón (LM 152)
    y el punto más alto visible de la cara (mínimo y entre los landmarks).
    """
    lms = face_landmarks.landmark

    # Mentón
    chin = lms[CHIN_IDX]
    y_chin = chin.y * h

    # Punto más alto: el landmark con y normalizada más pequeña
    top_idx = min(range(len(lms)), key=lambda i: lms[i].y)
    y_top = lms[top_idx].y * h

    # Distancia vertical en píxeles
    dist_px = abs(y_chin - y_top)
    return float(dist_px)


def medir_altura_menton_en_imagenes(image_paths: List[str]) -> List[Union[float, str]]:
    """
    Procesa una lista de paths a imágenes locales y devuelve, para cada una,
    la distancia (px) del mentón al punto más alto. Si no hay rostro, 'No face detected'.
    """
    results: List[Union[float, str]] = []

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,      # ideal para imágenes sueltas
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for path in tqdm(image_paths, desc="Midiendo (mentón→tope)", unit="imagen"):
            try:
                img = cv2.imread(path)
                if img is None:
                    results.append("Archivo no legible")
                    continue

                h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                out = face_mesh.process(rgb)

                if not out.multi_face_landmarks:
                    results.append("No face detected")
                    continue

                face_lms = out.multi_face_landmarks[0]
                dist_px = _chin_to_top_distance_px_from_landmarks(face_lms, w, h)
                results.append(round(dist_px, 2))
            except Exception as e:
                results.append(f"Error: {e}")

    return results


# ----------------------- Util: listar imágenes locales -----------------------

def listar_imagenes_locales(root_dir: str,
                            extensiones: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
                           ) -> List[str]:
    """
    Recorre recursivamente root_dir y retorna la lista de imágenes encontradas.
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"La ruta no existe o no es carpeta: {root_dir}")

    image_paths: List[str] = []
    for base, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(extensiones):
                image_paths.append(os.path.join(base, fn))

    image_paths.sort()
    return image_paths


# ----------------------- Pipeline LOCAL + Excel -----------------------

def process_local_folder_altura_menton(
    local_folder_path: str,
    forzar_descarga: bool = False,   # se ignora; lo mantenemos para compatibilidad de firma
) -> Tuple[List[str], List[Any]]:
    """
    Lee imágenes desde una carpeta LOCAL (ya montada en Colab), mide la distancia mentón→tope
    y retorna (paths_locales, distancias_o_mensajes).
    """
    print("[INFO] 🚀 Iniciando procesamiento LOCAL (altura mentón → punto más alto)...")

    # 1) Listar imágenes locales (sin API de Drive, sin descargas)
    image_paths = listar_imagenes_locales(local_folder_path)
    if not image_paths:
        print("[ERROR] No se encontraron imágenes en la carpeta local.")
        return [], []
    print(f"[INFO] Encontradas {len(image_paths)} imágenes en la carpeta local.")

    # 2) (Opcional) Validación/borrado de archivos corruptos usando tu helper existente
    process_image_list(image_paths)

    # 3) Medición principal
    print(f"[INFO] ✅ Listas {len(image_paths)} imágenes para medir distancia mentón→tope.")
    distances = medir_altura_menton_en_imagenes(image_paths)

    # 4) Pequeñas estadísticas
    nums = [d for d in distances if isinstance(d, (int, float))]
    no_face = sum(1 for d in distances if isinstance(d, str) and "No face" in d)
    errores = len(distances) - len(nums) - no_face
    if distances:
        print("\n[INFO] 📈 Estadísticas:")
        print(f" • Medidas válidas: {len(nums)}")
        print(f" • Sin rostro:      {no_face}")
        print(f" • Errores:         {errores}")
        if nums:
            arr = np.array(nums, dtype=float)
            print(f" • Promedio (px):   {arr.mean():.2f}")
            print(f" • Mín/Máx (px):    {arr.min():.2f} / {arr.max():.2f}")

    return image_paths, distances


# ----------------------- Main -----------------------

if __name__ == "__main__":
    # --- CONFIGURACIÓN: carpeta LOCAL ya montada en Colab ---
    dataset_local_path = (
        "/content/drive/MyDrive/INGENIERIA_EN_SOFTWARE/5to_Semestre/"
        "PRACTICAS/Primera_Revision/"
        "validator/results/sin_procesar"
    )
    os.makedirs("results", exist_ok=True)

    try:
        # Ejecutar pipeline LOCAL (sin descargas por API)
        paths, distances = process_local_folder_altura_menton(
            dataset_local_path,
            forzar_descarga=False,   # no aplica; se deja para compatibilidad
        )

        if not paths:
            sys.exit(1)

        # ---- Excel: 2 columnas (Ruta, Distancia_px) ----
        info = {
            "Ruta": format_to_hyperlinks(paths),
            "Distancia_menton_a_punto_mas_alto_px": distances,
        }
        normalized = normalize_dict_lengths(info)

        output_path = (
            f"/content/drive/MyDrive/colab/"
            f"Reporte_AlturaMenton_{get_file_count('results') + 1}.xlsx"
        )
        out = dict_to_excel(normalized, output_path)

        print(f"✅ ¡Listo! Reporte de Excel generado en: {out}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido por el usuario.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR FATAL] Un error inesperado ocurrió: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
