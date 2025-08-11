# modulos/puntos_faciales.py
# Reqs (ejemplo): pip install mediapipe opencv-python

from __future__ import annotations
import os, sys, time, tempfile, urllib.request, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from collections import deque
import statistics as stats

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ───────────────────────────── Config ─────────────────────────────

DEFAULT_MODEL_URLS = [
    # latest y un fallback “pinned”
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
]

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "0")  # mostrar logs de delegados (opcional)


@dataclass
class AppConfig:
    model_path: Path
    model_urls: list[str]
    min_bytes: int = 1_000_000
    delegate_preference: str = "gpu"  # auto | gpu | cpu
    camera_index: int = 0
    window_title: str = "Face Landmarker (GPU si disponible)"
    draw_landmarks: bool = True
    max_faces: int = 1
    min_face_detection_confidence: float = 0.5
    running_mode: vision.RunningMode = vision.RunningMode.VIDEO  # IMAGE | VIDEO | LIVE_STREAM


# ─────────────────────── Utilidades de modelo ───────────────────────

def ensure_file(path: Path, urls: Iterable[str], min_bytes: int = 1_000_000) -> Path:
    """Descarga a `path` si no existe o parece truncado."""
    if path.exists() and path.stat().st_size >= min_bytes:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for url in urls:
        try:
            print(f"Descargando modelo desde:\n  {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as r, \
                 tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent)) as tmp:
                total = int(r.headers.get("Content-Length") or 0)
                read = 0
                while True:
                    chunk = r.read(256 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
                    read += len(chunk)
                    if total:
                        pct = int(read * 100 / total)
                        print(f"\r  {read}/{total} bytes ({pct}%)", end="")
            print()
            tmp_path = Path(tmp.name)

            if tmp_path.stat().st_size < min_bytes:
                tmp_path.unlink(missing_ok=True)
                raise IOError("El archivo descargado parece demasiado pequeño.")

            tmp_path.replace(path)
            print(f"Modelo guardado en: {path}\n")
            return path

        except Exception as e:
            last_err = e
            print(f"  Falló con {e!r}. Probando siguiente URL...\n")

    raise FileNotFoundError(
        f"No se pudo descargar el modelo a {path}. Último error: {last_err}. "
        f"Exporta FACE_LANDMARKER_PATH apuntando a un .task válido para omitir la descarga."
    )


# ─────────────────────── Landmarker (GPU→CPU) ───────────────────────

class LandmarkerFactory:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def _create(self, use_gpu: bool):
        delegate = python.BaseOptions.Delegate.GPU if use_gpu else python.BaseOptions.Delegate.CPU
        base_opts = python.BaseOptions(model_asset_path=str(self.cfg.model_path), delegate=delegate)
        opts = vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=self.cfg.running_mode,
            num_faces=self.cfg.max_faces,
            min_face_detection_confidence=self.cfg.min_face_detection_confidence,
        )
        return vision.FaceLandmarker.create_from_options(opts)

    def create_with_fallback(self):
        pref = self.cfg.delegate_preference.lower()
        if pref == "cpu":
            lm = self._create(use_gpu=False)
            print("Face Landmarker: delegado CPU habilitado")
            return lm

        if pref == "gpu":
            lm = self._create(use_gpu=True)
            print("Face Landmarker: delegado GPU habilitado")
            return lm

        # auto: intentar GPU y caer a CPU si falla
        try:
            lm = self._create(use_gpu=True)
            print("Face Landmarker: delegado GPU habilitado")
            return lm
        except Exception as gpu_err:
            print(f"Delegado GPU falló ({gpu_err}). Cambiando a CPU…")
            lm = self._create(use_gpu=False)
            print("Face Landmarker: delegado CPU habilitado")
            return lm


# ─────────────────────── Métricas de rendimiento ───────────────────────

@dataclass
class PerfSnapshot:
    fps: float
    infer_ms: float
    e2e_ms: float


class PerfMeter:
    """Mide por segundo y también agrega métricas globales ponderadas por frame."""
    def __init__(self, warmup_sec: float = 2.0):
        self.last_report_t = time.perf_counter()
        self.frames_since = 0
        self.sum_infer_ms_win = 0.0
        self.sum_e2e_ms_win = 0.0
        self.last_snapshot = PerfSnapshot(0.0, 0.0, 0.0)

        # Globales
        self.start_t = self.last_report_t
        self.total_frames = 0
        self.total_infer_ms = 0.0
        self.total_e2e_ms = 0.0

        # Warm-up
        self.warmup_sec = warmup_sec
        self._warmup_done = False

        # Buffers para percentiles (opcional)
        self._infer_samples = deque(maxlen=60_000)  # ~muchos frames
        self._e2e_samples = deque(maxlen=60_000)

    def push(self, infer_ms: float, e2e_ms: float) -> PerfSnapshot:
        now = time.perf_counter()

        # Ventana de 1s para logs en vivo
        self.frames_since += 1
        self.sum_infer_ms_win += infer_ms
        self.sum_e2e_ms_win += e2e_ms

        # Warm-up gate
        if not self._warmup_done and (now - self.start_t) >= self.warmup_sec:
            # a partir de aquí contamos globales
            self._warmup_done = True

        if self._warmup_done:
            # Globales ponderadas por frame
            self.total_frames += 1
            self.total_infer_ms += infer_ms
            self.total_e2e_ms += e2e_ms
            # Para percentiles
            self._infer_samples.append(infer_ms)
            self._e2e_samples.append(e2e_ms)

        elapsed = now - self.last_report_t
        if elapsed >= 1.0:
            fps = self.frames_since / elapsed
            avg_inf = self.sum_infer_ms_win / self.frames_since
            avg_e2e = self.sum_e2e_ms_win / self.frames_since
            self.last_snapshot = PerfSnapshot(fps, avg_inf, avg_e2e)

            print(f"FPS: {fps:.1f} | infer(avg): {avg_inf:.1f} ms | e2e(avg): {avg_e2e:.1f} ms")

            # reset ventana 1s
            self.frames_since = 0
            self.sum_infer_ms_win = 0.0
            self.sum_e2e_ms_win = 0.0
            self.last_report_t = now

        return self.last_snapshot

    def summary(self) -> dict:
        end_t = time.perf_counter()
        dur = max(1e-9, end_t - self.start_t - self.warmup_sec)  # sin warm-up

        if self.total_frames == 0:
            return {
                "frames": 0,
                "duration_s": max(0.0, dur),
                "fps_global": 0.0,
                "infer_ms_global": float("nan"),
                "e2e_ms_global": float("nan"),
                "infer_ms_p50": float("nan"),
                "infer_ms_p90": float("nan"),
                "e2e_ms_p50": float("nan"),
                "e2e_ms_p90": float("nan"),
            }

        fps_global = self.total_frames / dur
        infer_ms_global = self.total_infer_ms / self.total_frames
        e2e_ms_global = self.total_e2e_ms / self.total_frames

        def pct(vals, q):
            if not vals: return float("nan")
            return float(stats.quantiles(vals, n=100, method="inclusive")[q-1])

        return {
            "frames": self.total_frames,
            "duration_s": dur,
            "fps_global": fps_global,
            "infer_ms_global": infer_ms_global,
            "e2e_ms_global": e2e_ms_global,
            "infer_ms_p50": stats.median(self._infer_samples) if self._infer_samples else float("nan"),
            "infer_ms_p90": pct(list(self._infer_samples), 90),
            "e2e_ms_p50": stats.median(self._e2e_samples) if self._e2e_samples else float("nan"),
            "e2e_ms_p90": pct(list(self._e2e_samples), 90),
        }


# ─────────────────────── Dibujo y overlay ───────────────────────

def draw_landmarks_bgr(frame_bgr, result) -> None:
    """Dibuja landmarks del primer rostro (si existe) sobre frame BGR."""
    if not result or not result.face_landmarks:
        return
    h, w = frame_bgr.shape[:2]
    # Por simplicidad, dibujamos SOLO el primer rostro (puedes iterar todos si max_faces>1)
    for lm in result.face_landmarks[0]:
        x = int(lm.x * w); y = int(lm.y * h)
        cv2.circle(frame_bgr, (x, y), 1, (0, 255, 0), -1)


def put_overlay(frame_bgr, snap: PerfSnapshot) -> None:
    txt = f"FPS {snap.fps:.1f} | infer {snap.infer_ms:.1f} ms | e2e {snap.e2e_ms:.1f} ms"
    cv2.putText(frame_bgr, txt, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


# ─────────────────────── Bucle principal ───────────────────────

def run_webcam(cfg: AppConfig) -> None:
    # Preparar modelo (descarga si hace falta)
    ensure_file(cfg.model_path, cfg.model_urls, cfg.min_bytes)

    # Crear landmarker con GPU→CPU (según preferencia)
    landmarker = LandmarkerFactory(cfg).create_with_fallback()

    # Abre con backend V4L2 (Linux). Si ya abriste, reabre así:
    cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara index={cfg.camera_index}")

    # Pide formato + resolución + FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))  # o 'MJPG' si tu cámara lo soporta
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # <- sí, úsalo

    # Verifica lo aplicado por el driver
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Solicitado 640x480@30 → Aplicado {int(w)}x{int(h)}@{fps:.2f} FPS")

    perf = PerfMeter()
    t0 = time.perf_counter()  # base para timestamps de VIDEO (ms crecientes)

    try:
        while True:
            t_frame_start = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                print("Fin de cámara / no hay frames.")
                break

            ts_ms = int((time.perf_counter() - t0) * 1000.0)  # ms crecientes para VIDEO

            # MediaPipe Tasks espera RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Inferencia
            t_inf_start = time.perf_counter()
            result = landmarker.detect_for_video(mp_image, ts_ms)
            t_inf_end = time.perf_counter()
            infer_ms = (t_inf_end - t_inf_start) * 1000.0

            # Dibujo
            if cfg.draw_landmarks:
                draw_landmarks_bgr(frame, result)

            # E2E
            t_frame_end = time.perf_counter()
            e2e_ms = (t_frame_end - t_frame_start) * 1000.0

            # Métricas por segundo
            snap = perf.push(infer_ms, e2e_ms)

            # Overlay
            put_overlay(frame, snap)

            cv2.imshow(cfg.window_title, frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        summary = perf.summary()
        print("\n=== Resumen global (sin warm-up) ===")
        print(f"Frames: {summary['frames']} | Duración: {summary['duration_s']:.2f} s")
        print(f"FPS global: {summary['fps_global']:.2f}")
        print(f"Infer: {summary['infer_ms_global']:.2f} ms (P50 {summary['infer_ms_p50']:.2f}, P90 {summary['infer_ms_p90']:.2f})")
        print(f"E2E  : {summary['e2e_ms_global']:.2f} ms (P50 {summary['e2e_ms_p50']:.2f}, P90 {summary['e2e_ms_p90']:.2f})")


# ─────────────────────── CLI ───────────────────────

def build_cfg_from_args() -> AppConfig:
    here = Path(__file__).resolve().parent
    root = here.parent if here.name == "tests" else here
    default_model = root / "models" / "face_landmarker.task"
    default_model = Path(os.getenv("FACE_LANDMARKER_PATH", str(default_model)))

    p = argparse.ArgumentParser(description="MediaPipe Face Landmarker (modular)")
    p.add_argument("--model", type=Path, default=default_model, help="Ruta del .task")
    p.add_argument("--camera", type=int, default=0, help="Índice de cámara (default: 0)")
    p.add_argument("--delegate", choices=["auto", "gpu", "cpu"], default="auto",
                   help="Preferencia de delegado TFLite (default: auto)")
    p.add_argument("--max-faces", type=int, default=1)
    p.add_argument("--min-conf", type=float, default=0.5, help="Min face detection confidence")
    p.add_argument("--no-draw", action="store_true", help="No dibujar landmarks")
    p.add_argument("--title", type=str, default="Face Landmarker (GPU si disponible)")

    args = p.parse_args()

    # Asegura carpeta del modelo
    args.model.parent.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        model_path=args.model,
        model_urls=list(DEFAULT_MODEL_URLS),
        delegate_preference=args.delegate,
        camera_index=args.camera,
        window_title=args.title,
        draw_landmarks=not args.no_draw,
        max_faces=args.max_faces,
        min_face_detection_confidence=args.min_conf,
    )


def main():
    try:
        cfg = build_cfg_from_args()
        run_webcam(cfg)
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
