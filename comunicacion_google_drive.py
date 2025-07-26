#!/usr/bin/env python3
"""
Descarga una imagen desde Google Drive y la muestra con matplotlib.
Requiere:
    pip install google-auth google-auth-oauthlib google-api-python-client pillow matplotlib
"""

import io
import pathlib

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from PIL import Image
import matplotlib.pyplot as plt

# ── 1. Autenticación ───────────────────────────────────────────────────────────
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_FILE = pathlib.Path("token.json")
CREDS_FILE = pathlib.Path("credentials.json")   # tu archivo OAuth 2.0

if TOKEN_FILE.exists():
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
else:
    flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN_FILE.write_text(creds.to_json())

drive = build("drive", "v3", credentials=creds)

# ── 2. Buscar el archivo (primer match) ────────────────────────────────────────
FILENAME = "resultado_analisis_chico.jpg"
query = f"name='{FILENAME}' and trashed=false"
results = drive.files().list(
    q=query,
    spaces="drive",
    fields="files(id, name)",
    pageSize=1,
).execute()

if not results["files"]:
    raise FileNotFoundError(f"No se encontró '{FILENAME}' en tu Google Drive.")

file_id = results["files"][0]["id"]

# ── 3. Descargar ───────────────────────────────────────────────────────────────
request = drive.files().get_media(fileId=file_id)
buffer = io.BytesIO()
downloader = MediaIoBaseDownload(buffer, request)

done = False
while not done:
    status, done = downloader.next_chunk()  # puedes imprimir status.progress() si quieres

buffer.seek(0)
image = Image.open(buffer)

# ── 4. Mostrar con matplotlib (funciona en entornos sin visor gráfico) ────────
plt.imshow(image)
plt.axis("off")         # quita ejes
plt.title(FILENAME)
plt.tight_layout()
plt.show()
