# pip install mediapipe opencv-python
import os, sys, tempfile, urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ───────────────────────── Config ─────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # show delegate logs (optional)

# Where to store/download the model (repo_root/models/face_landmarker.task)
HERE  = Path(__file__).resolve().parent
ROOT  = HERE.parent if HERE.name == "tests" else HERE
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "face_landmarker.task"

# Allow override via env var if you want to point elsewhere
MODEL_PATH = Path(os.getenv("FACE_LANDMARKER_PATH", str(MODEL_PATH)))

# Official URLs (try latest, then a pinned version as fallback)
MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
]

MIN_BYTES = 1_000_000  # basic sanity check (≈1 MB)


def ensure_file(path: Path, urls: list[str]) -> Path:
    """Download to `path` if missing or obviously truncated."""
    if path.exists() and path.stat().st_size >= MIN_BYTES:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in urls:
        try:
            print(f"Downloading model from:\n  {url}")
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})) as r, \
                 tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent)) as tmp:
                total = int(r.headers.get("Content-Length") or 0)
                read = 0
                while True:
                    chunk = r.read(1024 * 256)
                    if not chunk:
                        break
                    tmp.write(chunk)
                    read += len(chunk)
                    if total:
                        pct = int(read * 100 / total)
                        print(f"\r  {read}/{total} bytes ({pct}%)", end="")
                print()
                tmp.flush()
                tmp_name = tmp.name

            # simple size check
            if Path(tmp_name).stat().st_size < MIN_BYTES:
                Path(tmp_name).unlink(missing_ok=True)
                raise IOError("Downloaded file seems too small.")

            Path(tmp_name).replace(path)  # atomic move
            print(f"Saved model to: {path}\n")
            return path

        except Exception as e:
            last_err = e
            print(f"  Failed with {e!r}. Trying next URL...\n")

    raise FileNotFoundError(
        f"Could not download model to {path}. "
        f"Last error: {last_err}. "
        f"Set FACE_LANDMARKER_PATH to an existing file to skip download."
    )


# ─────────────────────── Prepare model & task ───────────────────────
MODEL_PATH = ensure_file(MODEL_PATH, MODEL_URLS)

# Try GPU first; if it blows up (e.g., EGL on WSL), fall back to CPU automatically.
def make_landmarker(use_gpu: bool = True):
    delegate = python.BaseOptions.Delegate.GPU if use_gpu else python.BaseOptions.Delegate.CPU
    base_opts = python.BaseOptions(model_asset_path=str(MODEL_PATH), delegate=delegate)
    opts = vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.VIDEO,  # IMAGE | VIDEO | LIVE_STREAM
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(opts)

try:
    landmarker = make_landmarker(use_gpu=True)
    print("Face Landmarker: GPU delegate enabled")
except Exception as gpu_err:
    print(f"GPU delegate failed ({gpu_err}). Falling back to CPU.")
    landmarker = make_landmarker(use_gpu=False)
    print("Face Landmarker: CPU delegate enabled")

# ───────────────────────── Webcam loop ─────────────────────────
cap = cv2.VideoCapture(0)
ts_ms = 0
window = "GPU Face Landmarker"  # label only; may be CPU if fallback happened

while True:
    ok, frame = cap.read()
    if not ok:
        break
    ts_ms += 33  # fake timestamp for VIDEO mode

    # MediaPipe Tasks expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, ts_ms)

    # draw simple markers
    if result.face_landmarks:
        for lm in result.face_landmarks[0]:
            x = int(lm.x * frame.shape[1]); y = int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow(window, frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
