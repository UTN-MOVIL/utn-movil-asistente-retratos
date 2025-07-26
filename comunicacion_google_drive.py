import io, pathlib
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image

# ----- 1. Authenticate (stores a token so you sign in once) -----
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
token_file = "token.json"

if pathlib.Path(token_file).exists():
    creds = Credentials.from_authorized_user_file(token_file, SCOPES)
else:
    flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
    creds = flow.run_local_server(port=0)
    pathlib.Path(token_file).write_text(creds.to_json())

drive = build('drive', 'v3', credentials=creds)

# Option B: search by name (first match)
results = drive.files().list(q="name='resultado_analisis_chico.jpg' and trashed=false",
                             spaces='drive',
                             fields='files(id, name)').execute()
file_id = results['files'][0]['id']

# ----- 3. Download -----
request = drive.files().get_media(fileId=file_id)
fh = io.BytesIO()
downloader = MediaIoBaseDownload(fh, request)
done = False
while not done:
    status, done = downloader.next_chunk()

# ----- 4. Use the image in Python -----
fh.seek(0)
img = Image.open(fh)
img.show()          # or img.save('local_copy.jpg')
