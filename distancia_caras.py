#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
from typing import List, Tuple, Any, Union
from tqdm import tqdm

import cv2
import mediapipe as mp
import numpy as np

from modulos.preprocesamiento import (
    drive_service,
    get_folder_id_by_path,
    list_files_recursive,
    download_files_parallel,
    process_image_list,
)

from modulos.exportacion_datos_excel import (
    format_to_hyperlinks,
    normalize_dict_lengths,
    dict_to_excel,
    get_file_count,
)

# Si quieres “reutilizar” explícitamente el módulo, puedes importarlo así;
# no es obligatorio para que funcione, pero lo dejo por claridad.
# from modulos import puntos_faciales as pf

CACHE_DIR = "image_cache"

# ----------------------- Cálculo con MediaPipe -----------------------

# Índice de mentón (MediaPipe Face Mesh)
CHIN_IDX = 152

def _chin_to_top_distance_px_from_landmarks(face_landmarks, w: int, h: int) -> float:
    """
    Devuelve la distancia euclidiana en píxeles entre el mentón (LM 152)
    y el punto más alto visible de la cara (mínimo y en los landmarks).
    """
    lms = face_landmarks.landmark

    # Mentón
    chin = lms[CHIN_IDX]
    x_chin = chin.x * w
    y_chin = chin.y * h

    # Punto más alto: el landmark con y normalizada más pequeña
    top_idx = min(range(len(lms)), key=lambda i: lms[i].y)
    top_lm = lms[top_idx]
    x_top = top_lm.x * w
    y_top = top_lm.y * h

    # Distancia euclidiana en px
    dist_px = abs(y_chin - y_top)
    return float(dist_px)


def medir_altura_menton_en_imagenes(image_paths: List[str]) -> List[Union[float, str]]:
    """
    Procesa una lista de paths a imágenes y devuelve, para cada una,
    la distancia (px) del mentón al punto más alto. Si no hay rostro, retorna 'No face detected'.
    """
    results: List[Union[float, str]] = []

    # Reutilizamos UNA sola instancia de FaceMesh para todas las imágenes
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
                # Si algo explota con esta imagen, marcamos el error
                results.append(f"Error: {e}")

    return results


# ----------------------- Pipeline Drive + Excel -----------------------

def process_drive_folder_altura_menton(
    drive_folder_path: str,
    max_workers: int = 4,
    forzar_descarga: bool = False,
) -> Tuple[List[str], List[Any]]:
    """
    Descarga/usa caché de imágenes de Drive, mide la distancia mentón→tope,
    y retorna (paths_locales, distancias_o_mensajes).
    """
    print("[INFO] 🚀 Iniciando procesamiento (altura mentón → punto más alto)...")

    drive = drive_service(force_reauth=False)
    folder_id = get_folder_id_by_path(drive_folder_path, drive)

    print("[INFO] Obteniendo lista de archivos remotos de Google Drive...")
    remote_files = list_files_recursive(folder_id, drive)
    if not remote_files:
        print("[ERROR] No se encontraron archivos en la carpeta de Drive.")
        return [], []
    print(f"[INFO] Encontrados {len(remote_files)} archivos en Drive.")

    # --- LÓGICA DE CACHÉ ---
    os.makedirs(CACHE_DIR, exist_ok=True)
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    remote_image_files = [
        (fid, name)
        for fid, name in remote_files
        if any(name.lower().endswith(ext) for ext in valid_ext)
    ]

    if forzar_descarga:
        print("[INFO] ⚠️ Forzando nueva descarga. Limpiando caché local...")
        for f in os.listdir(CACHE_DIR):
            try:
                os.remove(os.path.join(CACHE_DIR, f))
            except OSError as e:
                print(f"[WARNING] No se pudo eliminar {f} del caché: {e}")

    files_to_download = []
    cached_image_paths = []
    print("[INFO] 🔎 Verificando caché local...")
    for file_id, file_name in remote_image_files:
        local_path = os.path.join(CACHE_DIR, file_name)
        if os.path.exists(local_path) and not forzar_descarga:
            cached_image_paths.append(local_path)
        else:
            files_to_download.append((file_id, file_name))

    if cached_image_paths:
        print(f"[INFO] ✅ {len(cached_image_paths)} archivos encontrados en el caché.")

    if files_to_download:
        print(f"[INFO] 📥 Se descargarán {len(files_to_download)} archivos nuevos o faltantes.")
        downloaded_paths = download_files_parallel(
            files_to_download, CACHE_DIR, drive_service, max_workers
        )
        image_paths = cached_image_paths + downloaded_paths
    else:
        print("[INFO] ✅ El caché local ya está completo. No se necesitan descargas.")
        image_paths = cached_image_paths

    image_paths.sort()

    if not image_paths:
        print("[WARNING] No hay imágenes válidas para procesar.")
        return [], []

    # (Opcional) cualquier preprocesado que ya tengas
    process_image_list(image_paths)

    print(f"[INFO] ✅ Listas {len(image_paths)} imágenes para medir distancia mentón→tope.")

    # --- Medición principal ---
    distances = medir_altura_menton_en_imagenes(image_paths)

    # Pequeñas estadísticas
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
    # --- CONFIGURACIÓN ---
    dataset_drive_path = (
        "/Mi unidad/INGENIERIA_EN_SOFTWARE/5to_Semestre/"
        "PRACTICAS/Primera_Revision/"
        "validator/results/sin_procesar"
    )
    MAX_THREADS = 6

    os.makedirs("results", exist_ok=True)

    try:
        # Ejecutar pipeline de medición
        paths, distances = process_drive_folder_altura_menton(
            dataset_drive_path,
            max_workers=MAX_THREADS,
            forzar_descarga=False,   # pon True si quieres limpiar el caché
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
