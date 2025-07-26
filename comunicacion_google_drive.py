#!/usr/bin/env python3
"""
Cuenta el total de archivos (cualquier tipo) en la carpeta de Drive:
"/Mi unidad/INGENIERIA_EN_SOFTWARE/6to_Semestre/PRACTICAS/Practicas-FOTOS/
Primera_Revision/validator/results/validated_color"
"""

import os
import sys
import pathlib

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# ── 1. Autenticación ───────────────────────────────────────────────────────────
SCOPES     = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_FILE = pathlib.Path("token.json")
CREDS_FILE = pathlib.Path("credentials.json")

if TOKEN_FILE.exists():
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
else:
    flow  = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN_FILE.write_text(creds.to_json())

drive = build("drive", "v3", credentials=creds)

# ── 2. Obtener ID de carpeta por ruta ─────────────────────────────────────────
def get_folder_id_by_path(path):
    """
    De "/Mi unidad/Carp1/Carp2/..." obtiene recursivamente el ID de la última carpeta.
    """
    segments = [s for s in path.strip("/").split("/") if s and s != "Mi unidad"]
    parent_id = "root"
    for name in segments:
        resp = drive.files().list(
            q=(
                f"name = '{name}' "
                "and mimeType = 'application/vnd.google-apps.folder' "
                f"and '{parent_id}' in parents "
                "and trashed = false"
            ),
            spaces="drive",
            fields="files(id)",
            pageSize=1
        ).execute()
        items = resp.get("files", [])
        if not items:
            raise FileNotFoundError(f"Carpeta '{name}' no existe (parent={parent_id}).")
        parent_id = items[0]["id"]
    return parent_id

folder_path = (
    "/Mi unidad/INGENIERIA_EN_SOFTWARE/6to_Semestre/"
    "PRACTICAS/Practicas-FOTOS/Primera_Revision/"
    "validator/results/validated_color"
)
folder_id = get_folder_id_by_path(folder_path)

# ── 3. Contar todos los archivos en la carpeta ────────────────────────────────
page_token = None
total_files = 0

while True:
    resp = drive.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        spaces="drive",
        fields="nextPageToken, files(id)",
        pageToken=page_token
    ).execute()
    total_files += len(resp.get("files", []))
    page_token = resp.get("nextPageToken")
    if not page_token:
        break

# ── 4. Imprimir solo el total ─────────────────────────────────────────────────
print(total_files)
