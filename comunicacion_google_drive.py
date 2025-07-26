# %%capture
# ── Instala dependencias la primera vez (Drive, Pillow y Matplotlib) ──
# !pip install --quiet google-auth google-auth-oauthlib google-api-python-client pillow matplotlib

# ╔════════════════════════════════════════════════════════════════════╗
# ║  Descarga una imagen de Google Drive y la muestra dentro de Colab  ║
# ╚════════════════════════════════════════════════════════════════════╝

# ── Activa la salida ‘inline’ sólo si estamos en IPython/Jupyter ─────
import matplotlib.pyplot as plt
from IPython import get_ipython

shell = get_ipython()
if shell is not None:                       # Dentro de Colab/Jupyter
    shell.run_line_magic("matplotlib", "inline")  # equivale a %matplotlib inline

import io
import pathlib
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image

# ── 1. Autenticación ─────────────────────────────────────────────────
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

# ── 2. Buscar el archivo ─────────────────────────────────────────────
FILENAME = "resultado_analisis_chico.jpg"       # cámbialo a tu nombre
query = f"name='{FILENAME}' and trashed=false"
resp = drive.files().list(
    q=query, spaces="drive",
    fields="files(id, name)", pageSize=1
).execute()

if not resp["files"]:
    raise FileNotFoundError(f"No se encontró '{FILENAME}' en tu Google Drive.")

file_id = resp["files"][0]["id"]

# ── 3. Descargar el contenido ────────────────────────────────────────
buf = io.BytesIO()
downloader = MediaIoBaseDownload(buf, drive.files().get_media(fileId=file_id))
done = False
while not done:
    _, done = downloader.next_chunk()

buf.seek(0)
img = Image.open(buf)

# ── 4. Mostrar la imagen *inline* ────────────────────────────────────
plt.imshow(img)
plt.axis("off")          # quita los ejes
plt.show()
