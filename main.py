#!/usr/bin/env python3
import os
import io
import sys
import tempfile
from typing import List, Tuple

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from deteccion_lentes_v1 import get_glasses_probability
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
    """Del estilo '/Mi unidad/Carp1/Carp2' devuelve el ID de la última carpeta."""
    segments  = [s for s in path.strip("/").split("/") if s and s != "Mi unidad"]
    parent_id = "root"
    for name in segments:
        resp = (
            drive.files()
            .list(
                q=(
                    f"name = '{name}' and "
                    "mimeType = 'application/vnd.google-apps.folder' and "
                    f"'{parent_id}' in parents and trashed = false"
                ),
                fields="files(id)",
                pageSize=1,
            )
            .execute()
        )
        items = resp.get("files", [])
        if not items:
            raise FileNotFoundError(f"Carpeta '{name}' no encontrada (parent={parent_id})")
        parent_id = items[0]["id"]
    return parent_id

def list_files_recursive(folder_id: str, drive) -> List[Tuple[str, str]]:
    """
    Devuelve pares (file_id, drive_path) de todos los archivos (no carpetas)
    dentro de la carpeta indicada y sus subcarpetas.
    """
    results = []

    # primero listamos el contenido directo
    query = f"'{folder_id}' in parents and trashed = false"
    page_token = None
    while True:
        resp = (
            drive.files()
            .list(
                q=query,
                fields=(
                    "nextPageToken, "
                    "files(id, name, mimeType, parents)"
                ),
                pageToken=page_token,
            )
            .execute()
        )
        for f in resp["files"]:
            if f["mimeType"] == "application/vnd.google-apps.folder":
                # recursión en subcarpeta
                results.extend(list_files_recursive(f["id"], drive))
            else:
                results.append((f["id"], f["name"]))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results

def download_file(file_id: str, dest_path: str, drive):
    """Descarga un archivo de Drive al path local indicado."""
    request = drive.files().get_media(fileId=file_id)
    fh      = io.FileIO(dest_path, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()

# ── Procesamiento de la carpeta de Drive ─────────────────────────────────────
def process_drive_folder(drive_folder_path: str) -> Tuple[List[str], List[float]]:
    """
    Recorre la carpeta de Drive indicada, descarga los archivos y devuelve:
      - rutas locales de descarga (para el reporte)
      - probabilidades de lentes
    """
    drive   = drive_service()
    folder_id = get_folder_id_by_path(drive_folder_path, drive)

    files = list_files_recursive(folder_id, drive)
    if not files:
        print("No se encontraron archivos.")
        return [], []

    temp_dir = tempfile.mkdtemp(prefix="glasses_")
    image_paths: List[str] = []
    glasses_probs: List[float] = []

    for file_id, name in files:
        local_path = os.path.join(temp_dir, name)
        try:
            download_file(file_id, local_path, drive)
            prob = get_glasses_probability(local_path)
        except Exception as e:
            print(f"Saltando {name!r}: {e}")
            continue
        image_paths.append(local_path)
        glasses_probs.append(prob)

    return image_paths, glasses_probs

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Ruta dentro de Drive (respetar mayúsculas/minúsculas)
    dataset_drive_path = (
        "/Mi unidad/INGENIERIA_EN_SOFTWARE/6to_Semestre/"
        "PRACTICAS/Practicas-FOTOS/Primera_Revision/"
        "validator/results/validated_color"
    )

    results_folder = "results"
    os.makedirs(results_folder, exist_ok=True)

    paths, probs = process_drive_folder(dataset_drive_path)

    informacion = {
        "Rutas": format_to_hyperlinks(paths),
        "Probabilidad de tener lentes": probs,
    }

    normalized   = normalize_dict_lengths(informacion)
    output_file  = dict_to_excel(
        normalized,
        f"{results_folder}/Reporte_probabilidad_lentes_{get_file_count(results_folder)+1}.xlsx",
    )
    print(f"Excel generado en: {output_file}")
