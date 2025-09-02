#!/usr/bin/env python3
import os
import io
import threading
import concurrent.futures
from typing import List, Tuple

from tqdm import tqdm
from PIL import Image, ImageOps
import cv2
import numpy as np

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError


# ── Google Drive ──────────────────────────────────────────────────────────────
SCOPES     = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_FILE = "token.json"
CREDS_FILE = "credentials.json"

MAX_CACHE_SIZE = 100
_image_cache: dict[str, np.ndarray] = {}

# ───────────────────────── Utils: cache & resize ──────────────────────────────
def _get_image_hash(ruta_imagen: str) -> str:
    try:
        st = os.stat(ruta_imagen)
        return f"{ruta_imagen}_{st.st_mtime}_{st.st_size}"
    except FileNotFoundError:
        return ruta_imagen

def _load_image_optimized(ruta_imagen: str) -> np.ndarray:
    img_hash = _get_image_hash(ruta_imagen)

    if len(_image_cache) > MAX_CACHE_SIZE:
        # Evict 20 % oldest keys
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

# ─────────────────────── Google Drive helpers ────────────────────────────────
def drive_service(force_reauth: bool = False):
    # 1) Intenta ADC (funciona en Colab si ya corriste auth.authenticate_user())
    try:
        import google.auth
        creds, _ = google.auth.default(scopes=SCOPES)
        return build("drive", "v3", credentials=creds)
    except Exception as adc_err:
        print(f"[AUTH] ADC no disponible ({adc_err}). Probando flujo de app instalada…")

    # 2) Flujo de app instalada (token.json / credentials.json)
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.auth.exceptions import RefreshError

    TOKEN_FILE = "token.json"
    CREDS_FILE = "credentials.json"

    creds = None
    if os.path.exists(TOKEN_FILE) and not force_reauth:
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                try: os.remove(TOKEN_FILE)
                except OSError: pass
                creds = None

        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
            try:
                creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
            except Exception as e:
                print(f"[AUTH] run_local_server falló ({e}). Usando flujo por consola…")
                creds = flow.run_console()

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

# ⚡ OPT-2: chunk_size sube de 10 MB → 64 MB para reducir peticiones HTTP
def download_file_optimized(
    file_id: str,
    dest_path: str,
    drive,
    chunk_size: int = 64 * 1024 * 1024  # 64 MB  ✔
):
    request = drive.files().get_media(fileId=file_id)
    fh = io.FileIO(dest_path, "wb")
    downloader = MediaIoBaseDownload(fh, request, chunksize=chunk_size)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.close()

# ─────────────────────── Descarga paralela ───────────────────────────────────
thread_local = threading.local()

def download_files_parallel(
    files: List[Tuple[str, str]],
    temp_dir: str,
    drive_service_func,
    max_workers: int | None = None  # OPT-3: dinámico
) -> List[str]:
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    valid_files = [
        (fid, name) for fid, name in files
        if any(name.lower().endswith(ext) for ext in valid_ext)
    ]
    if not valid_files:
        print("[WARNING] No hay imágenes válidas para descargar")
        return []

    if max_workers is None:                         # OPT-3 ✔
        max_workers = min(8, len(valid_files))      # 6-8 hilos suele ser óptimo

    image_paths: list[str] = []
    errors = 0

    def get_thread_local_drive_service():
        if not hasattr(thread_local, 'drive'):
            print(f"[DEBUG] Creando nueva sesión Drive en hilo {threading.get_ident()}")
            thread_local.drive = drive_service_func()
        return thread_local.drive

    def _download_task(file_id: str, name: str):
        drive = get_thread_local_drive_service()
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

# ─────────────────────── Validación de imágenes ──────────────────────────────
def check_and_delete_corrupted_image(image_path: str) -> bool:
    try:
        _load_image_optimized(image_path)
        with Image.open(image_path).convert("RGB") as img:
            img.verify()
        return False
    except Exception:
        try:
            os.remove(image_path)
            print(f"[WARNING] Imagen corrupta o demasiado grande: {image_path}. Eliminada.")
        except OSError as e:
            print(f"[ERROR] No se pudo eliminar {image_path}: {e}")
        return True

def ajustar_imagen_375x425(ruta: str, destino: str | None = None):
    """
    Redimensiona a 375x425 px solo si hace falta. También normaliza orientación EXIF.
    Devuelve: (cambio_realizado: bool, ruta_salida: str, tamaño_final: (w,h))
    """
    with Image.open(ruta) as im:
        orig_format = im.format
        exif_obj = None
        try:
            exif_obj = im.getexif()
        except Exception:
            exif_obj = None
        orientation = exif_obj.get(274, 1) if exif_obj else 1  # 274 = Orientation

        im2 = ImageOps.exif_transpose(im)  # aplica orientación si es necesario

        changed = False
        if im2.size != (375, 425):
            im2 = im2.resize((375, 425), Image.Resampling.LANCZOS)
            changed = True

        # Si solo cambió la orientación (sin resize), también guardamos
        if not changed and orientation != 1:
            changed = True

        out = destino or ruta
        if changed or (destino and destino != ruta):
            if exif_obj:
                exif_obj[274] = 1  # deja la orientación “normalizada”
                exif_bytes = exif_obj.tobytes()
            else:
                exif_bytes = None

            save_kwargs = {}
            if exif_bytes:
                save_kwargs["exif"] = exif_bytes
            # Mantén el mismo formato si sobrescribes el mismo archivo
            if destino is None and orig_format:
                save_kwargs["format"] = orig_format

            im2.save(out, **save_kwargs)

        return changed, out, im2.size

def process_image_list(image_paths: List[str]) -> int:
    deleted_count = 0
    resized_count = 0
    skipped_count = 0
    errors = 0

    total_images = len(image_paths)
    print(f"🚀 Analizando {total_images} imágenes...")

    for path in image_paths:
        try:
            # Si se eliminó por corrupción, no continúes con ajustes
            if check_and_delete_corrupted_image(path):
                deleted_count += 1
                continue

            changed, _, _ = ajustar_imagen_375x425(path)
            if changed:
                resized_count += 1
            else:
                skipped_count += 1

        except FileNotFoundError:
            # El archivo pudo ser eliminado por otro proceso
            deleted_count += 1
        except Exception as e:
            errors += 1
            print(f"⚠️  Error procesando '{path}': {e}")

    print(f"\n✨ Listo: Revisadas {total_images} | Redimensionadas/ajustadas {resized_count} | "
          f"Saltadas {skipped_count} | Eliminadas {deleted_count} | Errores {errors}")
    return deleted_count