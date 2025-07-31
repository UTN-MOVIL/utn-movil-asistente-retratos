import os
from typing import List, Tuple
import io
import threading
import concurrent.futures
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
# ── Google Drive ──────────────────────────────────────────────────────────────
SCOPES     = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_FILE = "token.json"
CREDS_FILE = "credentials.json"

MAX_CACHE_SIZE = 100
_image_cache: dict[str, np.ndarray] = {}


def _get_image_hash(ruta_imagen: str) -> str:
    try:
        st = os.stat(ruta_imagen)
        return f"{ruta_imagen}_{st.st_mtime}_{st.st_size}"
    except FileNotFoundError:
        return ruta_imagen


def _load_image_optimized(ruta_imagen: str) -> np.ndarray:
    img_hash = _get_image_hash(ruta_imagen)

    if len(_image_cache) > MAX_CACHE_SIZE:
        for key in list(_image_cache)[: MAX_CACHE_SIZE // 5]:
            _image_cache.pop(key)

    if img_hash not in _image_cache:
        if not os.path.exists(ruta_imagen):
            raise FileNotFoundError(f"Imagen no encontrada: {ruta_imagen}")

        img = cv2.imread(ruta_imagen)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")

        h, w = img.shape[:2]
        target = 640
        if max(h, w) > target:
            scale = target / max(h, w)
            img = cv2.resize(
                img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
            )
            print(f"[INFO] Redimensionada {ruta_imagen} a ≤{target}px")

        _image_cache[img_hash] = img

    return _image_cache[img_hash]

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

thread_local = threading.local()

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

    def get_thread_local_drive_service():
        """Creates or retrieves a Drive service instance unique to the current thread."""
        if not hasattr(thread_local, 'drive'):
            # If this thread doesn't have a service object yet, create one
            print(f"[DEBUG] Creando nueva sesión de Drive para el hilo: {threading.get_ident()}")
            thread_local.drive = drive_service_func()
        # Return the service object for this thread
        return thread_local.drive

    def _download_task(file_id: str, name: str):
        """Downloads a single file using the thread-local Drive service."""
        # Get the single, persistent Drive service for this thread
        drive = get_thread_local_drive_service()
        local_path = os.path.join(temp_dir, name)
        try:
            download_file_optimized(file_id, local_path, drive)
            return local_path
        except Exception as e:
            # Using a thread-safe print or logging mechanism is better here if needed
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

def check_and_delete_corrupted_image(image_path: str) -> bool:
    """
    Checks if an image is corrupted or excessively large (Decompression Bomb).
    
    If an issue is found, it prints a warning, deletes the file, and returns True.
    Otherwise, it returns False.
    """
    try:
        _load_image_optimized(image_path)  # Attempt to load the image using the optimized function
        with Image.open(image_path) as img:
            img.verify()  # Checks for file integrity.
        return False  # Image is OK.

    # --- THIS IS THE CORRECTED PART ---
    except Exception:
        # Catch the specific error for oversized images
        try:
            os.remove(image_path)
            print(f"[WARNING] Imagen corrupta o demasiado grande: {image_path}. Eliminada.")
        except OSError as e:
            print(f"[ERROR] Could not delete file {image_path}: {e}")
        return True # Signal that the file was problematic and handled.
        
def process_image_list(image_paths: List[str]) -> int:
    """
    Processes a list of image paths, deleting any that are corrupted.

    Args:
        image_paths: A list of file paths to check.

    Returns:
        The total number of corrupted images that were deleted.
    """
    deleted_count = 0
    total_images = len(image_paths)
    
    print(f"🚀 Starting to process {total_images} images...")

    for i, path in enumerate(image_paths):
        # The print statement is now inside check_and_delete_corrupted_image
        if check_and_delete_corrupted_image(path):
            deleted_count += 1
            
    print(f"\n✨ Processing Complete!")
    print(f"Checked: {total_images} | Deleted: {deleted_count}")
    return deleted_count