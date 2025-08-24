# main.py
# Ejemplo que usa puntos faciales + (opcional) esqueleto/pose en el MISMO loop:
# 1) Descargar/cargar modelos
# 2) Leer webcam
# 3) Dibujar TODOS los puntos faciales (primer rostro) + esqueleto (todas las poses)
# 4) Mostrar métricas (FPS / infer totales) + desgloses por tarea

from __future__ import annotations
import os
import sys
import time
import argparse
from pathlib import Path

import cv2
import mediapipe as mp  # para mp.Image (MediaPipe Tasks)

# ── Rutas / módulo local ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]  # …/funcionalidades_validador_retratos
sys.path.insert(0, str(ROOT))

# --- Facial landmarks (tu módulo existente) ---
from modules.puntos_faciales import (
    AppConfig as FaceConfig,
    DEFAULT_MODEL_URLS as FACE_MODEL_URLS,
    ensure_file as ensure_face_file,
    LandmarkerFactory as FaceLandmarkerFactory,
    PerfMeter,                       # lo reutilizamos
    draw_landmarks_bgr as draw_face_landmarks_bgr,
    put_overlay as put_overlay_line1 # lo usamos para la primera línea
)

# --- Pose / esqueleto (nuevo) ---
from modules.esqueleto import (
    AppConfig as PoseConfig,
    DEFAULT_MODEL_URLS as POSE_MODEL_URLS,
    ensure_file as ensure_pose_file,
    LandmarkerFactory as PoseLandmarkerFactory,
    draw_pose_skeleton_bgr
)


def build_cfg_from_args():
    here = Path(__file__).resolve().parent
    # Modelos por defecto
    default_face_model = Path(os.getenv("FACE_LANDMARKER_PATH", str(here / "models" / "face_landmarker.task")))
    default_pose_model = Path(os.getenv("POSE_LANDMARKER_PATH", str(here / "models" / "pose_landmarker_full.task")))

    p = argparse.ArgumentParser(description="Dibujo de puntos faciales + (opcional) esqueleto/pose")
    # Cámara / ventana
    p.add_argument("--camera", type=int, default=0, help="Índice de cámara (default: 0)")
    p.add_argument("--title", type=str, default="Face + Pose (GPU si disponible)")
    p.add_argument("--w", type=int, default=640, help="Ancho solicitado")
    p.add_argument("--h", type=int, default=480, help="Alto solicitado")
    p.add_argument("--fps", type=int, default=30, help="FPS solicitados")

    # FACE
    p.add_argument("--model", type=Path, default=default_face_model, help="Ruta del .task de FACE")
    p.add_argument("--delegate", choices=["auto", "gpu", "cpu"], default="auto",
                   help="Preferencia delegado FACE (default: auto)")
    p.add_argument("--max-faces", type=int, default=1)
    p.add_argument("--min-conf", type=float, default=0.5, help="Min face detection confidence")

    # POSE (opcional)
    p.add_argument("--with-pose", action="store_true", help="También dibujar esqueleto/pose")
    p.add_argument("--pose-model", type=Path, default=default_pose_model, help="Ruta del .task de POSE")
    p.add_argument("--pose-delegate", choices=["auto", "gpu", "cpu"], default="auto",
                   help="Preferencia delegado POSE (default: auto)")
    p.add_argument("--pose-max", type=int, default=1, help="Cantidad máxima de poses")
    p.add_argument("--pose-min-conf", type=float, default=0.5, help="Min pose detection confidence")

    args = p.parse_args()
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.pose_model.parent.mkdir(parents=True, exist_ok=True)

    face_cfg = FaceConfig(
        model_path=args.model,
        model_urls=list(FACE_MODEL_URLS),
        delegate_preference=args.delegate,
        camera_index=args.camera,
        window_title=args.title,
        draw_landmarks=True,
        max_faces=args.max_faces,
        min_face_detection_confidence=args.min_conf,
    )

    pose_cfg = None
    if args.with_pose:
        # Nota: los nombres de campos difieren entre módulos (max_poses vs max_faces)
        pose_cfg = PoseConfig(
            model_path=args.pose_model,
            model_urls=list(POSE_MODEL_URLS),
            delegate_preference=args.pose_delegate,
            camera_index=args.camera,
            window_title=args.title,
            draw_landmarks=True,
            max_poses=args.pose_max,
            min_pose_detection_confidence=args.pose_min_conf,
        )

    return face_cfg, pose_cfg, int(args.w), int(args.h), int(args.fps)


def _put_overlay_line2(frame_bgr, face_ms: float | None, pose_ms: float | None, y: int = 46) -> None:
    """Segunda línea de overlay con desgloses de inferencia."""
    parts = []
    if face_ms is not None:
        parts.append(f"face {face_ms:.1f} ms")
    if pose_ms is not None:
        parts.append(f"pose {pose_ms:.1f} ms")
    if parts:
        txt = " | ".join(parts)
        cv2.putText(frame_bgr, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


def run_webcam_draw_points(face_cfg: FaceConfig,
                           pose_cfg: PoseConfig | None,
                           width: int, height: int, target_fps: int) -> None:
    # 1) Asegurar modelos
    ensure_face_file(face_cfg.model_path, face_cfg.model_urls, face_cfg.min_bytes)
    pose_landmarker = None
    if pose_cfg:
        ensure_pose_file(pose_cfg.model_path, pose_cfg.model_urls, pose_cfg.min_bytes)

    # 2) Crear landmarkers con fallback GPU→CPU
    face_landmarker = FaceLandmarkerFactory(face_cfg).create_with_fallback()
    if pose_cfg:
        pose_landmarker = PoseLandmarkerFactory(pose_cfg).create_with_fallback()

    # 3) Abrir cámara
    cap = cv2.VideoCapture(face_cfg.camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara index={face_cfg.camera_index}")

    # Solicitar formato + resolución + FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))  # o 'MJPG' si tu cámara lo soporta
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Solicitado {width}x{height}@{target_fps} → Aplicado {w}x{h}@{fps:.2f} FPS")

    perf = PerfMeter()
    t0 = time.perf_counter()  # base para timestamps de VIDEO (ms crecientes)

    try:
        while True:
            t_frame_start = time.perf_counter()

            ok, frame_bgr = cap.read()
            if not ok:
                print("Fin de cámara / no hay frames.")
                break

            # MediaPipe Tasks espera RGB
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamp creciente en ms (modo VIDEO)
            ts_ms = int((time.perf_counter() - t0) * 1000.0)

            # 4) Inferencias
            face_ms = pose_ms = None

            t_inf_start = time.perf_counter()
            face_result = face_landmarker.detect_for_video(mp_image, ts_ms)
            t_inf_end = time.perf_counter()
            face_ms = (t_inf_end - t_inf_start) * 1000.0

            if pose_landmarker:
                t_pose_start = time.perf_counter()
                pose_result = pose_landmarker.detect_for_video(mp_image, ts_ms)
                t_pose_end = time.perf_counter()
                pose_ms = (t_pose_end - t_pose_start) * 1000.0
            else:
                pose_result = None

            # 5) Dibujo
            draw_face_landmarks_bgr(frame_bgr, face_result)
            if pose_result is not None:
                draw_pose_skeleton_bgr(frame_bgr, pose_result)

            # 6) Métricas / overlay
            e2e_ms = (time.perf_counter() - t_frame_start) * 1000.0
            infer_total = (face_ms or 0.0) + (pose_ms or 0.0)
            snap = perf.push(infer_total, e2e_ms)

            # Primera línea (tu overlay original): FPS / infer total / e2e
            put_overlay_line1(frame_bgr, snap)
            # Segunda línea: desglose por tarea
            _put_overlay_line2(frame_bgr, face_ms, pose_ms)

            cv2.imshow(face_cfg.window_title, frame_bgr)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        summary = perf.summary()
        print("\n=== Resumen global (sin warm-up) ===")
        print(f"Frames: {summary['frames']} | Duración: {summary['duration_s']:.2f} s")
        print(f"FPS global: {summary['fps_global']:.2f}")
        print(f"Infer (total): {summary['infer_ms_global']:.2f} ms "
              f"(P50 {summary['infer_ms_p50']:.2f}, P90 {summary['infer_ms_p90']:.2f})")
        print(f"E2E  : {summary['e2e_ms_global']:.2f} ms "
              f"(P50 {summary['e2e_ms_p50']:.2f}, P90 {summary['e2e_ms_p90']:.2f})")


def main():
    try:
        face_cfg, pose_cfg, W, H, FPS = build_cfg_from_args()
        run_webcam_draw_points(face_cfg, pose_cfg, width=W, height=H, target_fps=FPS)
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()