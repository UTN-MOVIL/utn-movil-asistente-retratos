#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# distancia_caras.py

import os
import sys
from typing import List, Tuple, Any, Union
from tqdm import tqdm

import cv2
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

# Reutiliza el alias FM y las funciones del módulo común
from modulos.puntos_faciales import (
    FM,
    chin_to_top_distance_px_from_landmarks as chin_dist,
    porcentaje_rostro_desde_landmarks as pct_from_lms,
)

CACHE_DIR = "image_cache"   # ya no se usa para descargar, pero lo dejamos por compatibilidad


# ----------------------- Cálculo por imagen (1 pasada) -----------------------
def medir_altura_y_porcentaje_en_imagenes(
    image_paths: List[str],
    usar_convhull: bool = False
) -> Tuple[List[Union[float, str]], List[Union[float, str]]]:
    """
    Devuelve dos listas paralelas:
      - distancias_px: distancia mentón→punto más alto (px)
      - porcentajes:   % de rostro respecto al área de la imagen
    Si falla o no hay rostro, retorna el mismo mensaje en ambas listas.

    Hace UNA sola inferencia FaceMesh por imagen y reutiliza los mismos landmarks.
    """
    distancias: List[Union[float, str]] = []
    porcentajes: List[Union[float, str]] = []

    with FM.FaceMesh(
        static_image_mode=True,      # ideal para imágenes sueltas
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for path in tqdm(image_paths, desc="Midiendo (mentón→tope y % rostro)", unit="imagen"):
            try:
                img = cv2.imread(path)
                if img is None:
                    msg = "Archivo no legible"
                    distancias.append(msg)
                    porcentajes.append(msg)
                    continue

                h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                out = face_mesh.process(rgb)

                if not out.multi_face_landmarks:
                    msg = "No face detected"
                    distancias.append(msg)
                    porcentajes.append(msg)
                    continue

                face_lms = out.multi_face_landmarks[0]

                # Distancia (reutiliza tu función del módulo):
                dist_px = chin_dist(face_lms, w, h)
                # Porcentaje (desde los mismos landmarks):
                pct = pct_from_lms(face_lms, w, h, usar_convhull=usar_convhull)

                distancias.append(round(float(dist_px), 2))
                porcentajes.append(round(float(pct), 2))
            except Exception as e:
                msg = f"Error: {e}"
                distancias.append(msg)
                porcentajes.append(msg)

    return distancias, porcentajes


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
    usar_convhull: bool = False      # <- controla el % de rostro
) -> Tuple[List[str], List[Any], List[Any]]:
    """
    Lee imágenes desde una carpeta LOCAL, mide la distancia mentón→tope y el % de rostro.
    Retorna (paths_locales, distancias_o_mensajes, porcentajes_o_mensajes).
    """
    print("[INFO] 🚀 Iniciando procesamiento LOCAL (altura mentón → punto más alto + % rostro)...")

    # 1) Listar imágenes locales (sin API de Drive, sin descargas)
    image_paths = listar_imagenes_locales(local_folder_path)
    if not image_paths:
        print("[ERROR] No se encontraron imágenes en la carpeta local.")
        return [], [], []
    print(f"[INFO] Encontradas {len(image_paths)} imágenes en la carpeta local.")

    # 2) (Opcional) Validación/borrado de archivos corruptos usando tu helper existente
    process_image_list(image_paths)

    # 3) Medición principal (1 sola pasada)
    print(f"[INFO] ✅ Listas {len(image_paths)} imágenes para medir distancia y % rostro.")
    distances, percents = medir_altura_y_porcentaje_en_imagenes(
        image_paths,
        usar_convhull=usar_convhull
    )

    # 4) Pequeñas estadísticas
    nums_dist = [d for d in distances if isinstance(d, (int, float, np.floating))]
    nums_pct  = [p for p in percents  if isinstance(p, (int, float, np.floating))]
    no_face   = sum(1 for d in distances if isinstance(d, str) and "No face" in d)
    errores   = len(distances) - len(nums_dist) - no_face
    if distances:
        print("\n[INFO] 📈 Estadísticas:")
        print(f" • Medidas válidas:       {len(nums_dist)}")
        print(f" • Sin rostro:            {no_face}")
        print(f" • Errores:               {errores}")
        if nums_dist:
            arr = np.array(nums_dist, dtype=float)
            print(f" • Distancia px (prom):   {arr.mean():.2f}  min/max: {arr.min():.2f}/{arr.max():.2f}")
        if nums_pct:
            arr = np.array(nums_pct, dtype=float)
            print(f" • % rostro (prom):       {arr.mean():.2f}  min/max: {arr.min():.2f}/{arr.max():.2f}")

    return image_paths, distances, percents


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
        paths, distances, percents = process_local_folder_altura_menton(
            dataset_local_path,
            forzar_descarga=False,   # no aplica; se deja para compatibilidad
            usar_convhull=False      # cámbialo a True si prefieres convex hull
        )

        if not paths:
            sys.exit(1)

        # ---- Excel: 3 columnas (Ruta, Distancia_px, %_rostro) ----
        def _round_or_msg(x):
            return round(float(x), 2) if isinstance(x, (int, float, np.floating)) else x

        info = {
            "Ruta": format_to_hyperlinks(paths),
            "Distancia_menton_a_punto_mas_alto_px": [ _round_or_msg(d) for d in distances ],
            "Porcentaje_rostro": [ _round_or_msg(p) for p in percents ],
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
