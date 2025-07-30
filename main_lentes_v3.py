#!/usr/bin/env python3
import concurrent.futures
import os
import io
import sys
import threading
from queue import Queue
from typing import List, Tuple, Any
from tqdm import tqdm

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# --- MODIFICACIÓN: Importar el nuevo detector ---
# Se asume que el archivo deteccion_lentes_v3.py contiene tu función
# y todas sus dependencias necesarias (dlib, numpy, cv2, etc.)
from deteccion_lentes_v3 import glasses_detect

from exportacion_datos_excel import (
    format_to_hyperlinks,
    normalize_dict_lengths,
    dict_to_excel,
    get_file_count,
)

# ── Google Drive ──────────────────────────────────────────────────────────────
SCOPES     = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_FILE = "token.json"
CREDS_FILE = "credentials.json"
CACHE_DIR  = "image_cache"

# --- MODIFICACIÓN: El umbral de probabilidad ya no es necesario ---

def drive_service():
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    else:
        flow  = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return build("drive", "v3", credentials=creds)

def get_folder_id_by_path(path: str, drive):
    segments  = [s for s in path.strip("/").split("/") if s and s != "Mi unidad"]
    parent_id = "root"
    for name in segments:
        resp = drive.files().list(
            q=(
                f"name = '{name}' and "
                "mimeType = 'application/vnd.google-apps.folder' and "
                f"'{parent_id}' in parents and trashed = false"
            ),
            fields="files(id)",
            pageSize=1,
        ).execute()
        items = resp.get("files", [])
        if not items:
            raise FileNotFoundError(f"Carpeta '{name}' no encontrada (parent={parent_id})")
        parent_id = items[0]["id"]
    return parent_id

def list_files_recursive(folder_id: str, drive) -> List[Tuple[str, str]]:
    results = []
    query = f"'{folder_id}' in parents and trashed = false"
    page_token = None
    while True:
        resp = drive.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
        ).execute()
        for f in resp["files"]:
            if f["mimeType"] == "application/vnd.google-apps.folder":
                results.extend(list_files_recursive(f["id"], drive))
            else:
                results.append((f["id"], f["name"]))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results

def download_file_optimized(file_id: str, dest_path: str, drive, chunk_size: int = 10 * 1024 * 1024):
    request = drive.files().get_media(fileId=file_id)
    fh = io.FileIO(dest_path, "wb")
    downloader = MediaIoBaseDownload(fh, request, chunksize=chunk_size)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.close()

def download_files_parallel(
    files: List[Tuple[str, str]],
    temp_dir: str,
    drive_service_func,
    max_workers: int = 4
) -> List[str]:
    valid_ext = {'.jpg','.jpeg','.png','.bmp','.tiff','.webp'}
    valid_files = [
        (fid, name) for fid, name in files
        if any(name.lower().endswith(ext) for ext in valid_ext)
    ]
    if not valid_files:
        print("[WARNING] No hay imágenes válidas para descargar")
        return []

    image_paths = []
    errors = 0

    def _download_task(file_id: str, name: str):
        drive = drive_service_func()
        local_path = os.path.join(temp_dir, name)
        try:
            download_file_optimized(file_id, local_path, drive)
            return local_path
        except Exception as e:
            raise RuntimeError(f"Error al descargar '{name}': {e}") from e

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(_download_task, fid, name): name
            for fid, name in valid_files
        }
        progress_bar = tqdm(
            concurrent.futures.as_completed(future_to_name),
            total=len(valid_files),
            desc="Descargando",
            unit="archivo"
        )
        for future in progress_bar:
            try:
                path = future.result()
                image_paths.append(path)
            except Exception as e:
                print(f"[ERROR] {e}")
                errors += 1
            progress_bar.set_postfix(exitosos=len(image_paths), errores=errors)

    print(f"[INFO] Descarga completa: {len(image_paths)} éxitos, {errors} errores")
    return image_paths

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

    print(f"[INFO] ✅ Listas {len(image_paths)} imágenes para procesar.")
    
    # --- MODIFICACIÓN: FASE 2: Detección con glasses_detector ---
    print("[INFO] 🔍 Iniciando detección de lentes (método v2)...")
    detection_results: List[Any] = []
    for path in tqdm(image_paths, desc="Detectando lentes (v2)", unit="imagen"):
        try:
            # Llamada a la nueva función importada
            result = glasses_detect(path)
            detection_results.append(result)
        except Exception as e:
            # Captura cualquier error inesperado del detector
            print(f"[ERROR] Procesando {os.path.basename(path)}: {e}")
            detection_results.append('Error de procesamiento')

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
            if r == "present": return "SÍ"
            if r == "absent": return "NO"
            return str(r).replace('_', ' ').upper() # Formatea 'No face detected' y otros errores

        info = {
            "Rutas": format_to_hyperlinks(paths),
            "Resultado_Raw": results,
            "Deteccion_Lentes": [format_result(r) for r in results]
        }
        normalized = normalize_dict_lengths(info)
        
        output_path = f"results/Reporte_v2_{get_file_count('results')+1}.xlsx"
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