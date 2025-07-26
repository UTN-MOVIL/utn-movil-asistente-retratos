#!/usr/bin/env python3
"""
Descarga una imagen de Google Drive y la muestra:
• Si hay backend GUI ➜ ventana interactiva
• Si no ➜ la guarda como PNG y avisa la ruta
"""

import io, pathlib, os, sys
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

# ── Seleccionar backend ────────────────────────────────────────────────────────
if os.environ.get("DISPLAY", "") == "" and sys.platform != "win32":
    # Entorno sin servidor X → backend headless
    matplotlib.use("Agg")
    HEADLESS = True
else:
    # Hay GUI → intenta TkAgg (o el que tengas)
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    HEADLESS = False

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

# ── 2. Buscar el archivo ───────────────────────────────────────────────────────
FILENAME = "resultado_analisis_chico.jpg"
query = f"name='{FILENAME}' and trashed=false"
resp = drive.files().list(q=query, spaces="drive",
                          fields="files(id, name)", pageSize=1).execute()
if not resp["files"]:
    raise FileNotFoundError(f"No se encontró '{FILENAME}'.")

file_id = resp["files"][0]["id"]

# ── 3. Descargar el contenido ──────────────────────────────────────────────────
buf = io.BytesIO()
downloader = MediaIoBaseDownload(buf, drive.files().get_media(fileId=file_id))
done = False
while not done:
    _, done = downloader.next_chunk()

buf.seek(0)
img = Image.open(buf)

# ── 4. Mostrar o guardar según el backend ──────────────────────────────────────
plt.imshow(img)
plt.axis("off")
plt.title(FILENAME)
plt.tight_layout()

if HEADLESS:
    out_path = pathlib.Path("preview.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"No hay backend gráfico. Imagen guardada en {out_path.resolve()}")
else:
    plt.show(block=True)          # ventana interactiva
