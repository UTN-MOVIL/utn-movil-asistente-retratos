#!/usr/bin/env python3
import os
import sys
from queue import Queue
from typing import List, Tuple, Any
from tqdm import tqdm

from preprocesamiento import (
    drive_service,
    get_folder_id_by_path,
    list_files_recursive,
    download_files_parallel,
    process_image_list
)

# --- MODIFICACIÓN: Importar el nuevo detector ---
# Se asume que el archivo deteccion_lentes_v3.py contiene tu función
# y todas sus dependencias necesarias (dlib, numpy, cv2, etc.)
from deteccion_lentes_v1 import get_glasses_probability

from exportacion_datos_excel import (
    format_to_hyperlinks,
    normalize_dict_lengths,
    dict_to_excel,
    get_file_count,
)

CACHE_DIR  = "image_cache"

# ── Procesamiento principal con el nuevo detector ──────────────────────────
def process_drive_folder_with_detector_v2(
    drive_folder_path: str,
    max_workers: int = 4,
    forzar_descarga: bool = False
) -> Tuple[List[str], List[Any]]:
    """
    Procesa imágenes de Drive con el detector v2, usando caché local.
    """
    print("[INFO] 🚀 Iniciando procesamiento con detector v2...")
    
    drive = drive_service()
    folder_id = get_folder_id_by_path(drive_folder_path, drive)
    
    print("[INFO] Obteniendo lista de archivos remotos de Google Drive...")
    remote_files = list_files_recursive(folder_id, drive)
    if not remote_files:
        print("[ERROR] No se encontraron archivos en la carpeta de Drive.")
        return [], []
    print(f"[INFO] Encontrados {len(remote_files)} archivos en Drive.")
    
    # --- LÓGICA DE CACHÉ (sin cambios) ---
    os.makedirs(CACHE_DIR, exist_ok=True)
    valid_ext = {'.jpg','.jpeg','.png','.bmp','.tiff','.webp'}
    remote_image_files = [
        (fid, name) for fid, name in remote_files 
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

    process_image_list(image_paths)

    print(f"[INFO] ✅ Listas {len(image_paths)} imágenes para procesar.")
    
    # --- MODIFICACIÓN: FASE 2: Detección con get_glasses_probabilityor ---
    print("[INFO] 🔍 Iniciando detección de lentes...")
    detection_results: List[Any] = []

    for path in tqdm(image_paths, desc="Detectando lentes", unit="imagen"):
        try:
            # Intenta procesar la imagen como siempre
            result = get_glasses_probability(path)
            detection_results.append(result)

        except Exception as e:
            # SI OCURRE UN ERROR:
            # 1. Informa del problema y que el archivo será eliminado.
            print(f"\n[ERROR] Problema con '{os.path.basename(path)}': {e}. Eliminando archivo... 🗑️")
            
            # 2. Intenta eliminar el archivo del disco.
            try:
                os.remove(path)
            except OSError as oe:
                print(f"[WARNING] No se pudo eliminar el archivo {path}: {oe}")

            # 3. Registra el error en tus resultados para el reporte final.
            detection_results.append('Archivo corrupto y eliminado')

    # --- MODIFICACIÓN: FASE 3: Estadísticas finales adaptadas ---
    print("\n[INFO] 📈 Estadísticas finales:")
    total = len(detection_results)
    if total > 0:
        con_lentes = detection_results.count(1)
        sin_lentes = detection_results.count(0)
        sin_rostro = detection_results.count('No face detected')
        errores = total - (con_lentes + sin_lentes + sin_rostro)
        
        porc_con_lentes = (con_lentes / total) * 100
        porc_sin_lentes = (sin_lentes / total) * 100
        porc_sin_rostro = (sin_rostro / total) * 100

        print(f"👓 Con lentes: {con_lentes} ({porc_con_lentes:.1f}%)")
        print(f"👁️ Sin lentes: {sin_lentes} ({porc_sin_lentes:.1f}%)")
        print(f"❓ Sin rostro detectado: {sin_rostro} ({porc_sin_rostro:.1f}%)")
        if errores > 0:
            print(f"💥 Errores: {errores}")
    else:
        print("⚠️ No se procesaron imágenes. No se pueden calcular estadísticas.")

    return image_paths, detection_results

# ── Main ─────────────────────────────────────────────────────────────────────
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
        # --- MODIFICACIÓN: Llamar a la nueva función principal ---
        paths, results = process_drive_folder_with_detector_v2(
            dataset_drive_path,
            max_workers=MAX_THREADS
        )
        if not paths:
            sys.exit(1)
            
        # --- MODIFICACIÓN: Crear diccionario de resultados para el Excel ---
        def format_result(r):
            # First, check if 'r' is a number (float or int)
            if isinstance(r, (float, int)):
                if r >= 0.4486: return "SÍ"
                return "NO"  # No need for the second 'if', it's the only other case
            
            # If it's not a number, it must be a string. Handle it here.
            return str(r).replace('_', ' ').upper()

        info = {
            "Rutas": format_to_hyperlinks(paths),
            "Resultado_Raw": results,
            "Deteccion_Lentes": [format_result(r) for r in results]
        }
        normalized = normalize_dict_lengths(info)
        
        output_path = f"/content/drive/MyDrive/colab/Reporte_v2_{get_file_count('results')+1}.xlsx"
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