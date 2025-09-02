# connection/config.py
import os
import time
import traceback

def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")

# Usa "1" como default si así lo quieres (en tu pregunta lo pones en 1).
PRINT_LOGS = _env_bool("PRINT_LOGS", "1")
STUN_URL = os.getenv("STUN_URL", "stun:stun.l.google.com:19302")
TURN_URL = os.getenv("TURN_URL")
TURN_USER = os.getenv("TURN_USERNAME")
TURN_PASS = os.getenv("TURN_PASSWORD")

POSE_USE_VIDEO = os.getenv("POSE_USE_VIDEO", "0") == "1"
ABSOLUTE_INTERVAL_MS = int(os.getenv("ABSOLUTE_INTERVAL_MS", "0"))
IDLE_TO_FORCE_KF_MS = int(os.getenv("IDLE_TO_FORCE_KF_MS", "500"))
FRAME_GAP_WARN_MS = int(os.getenv("FRAME_GAP_WARN_MS", "180"))
RESULTS_REQUIRE_ACK = os.getenv("RESULTS_REQUIRE_ACK", "0") == "1"
ACK_WARN_MS = int(os.getenv("ACK_WARN_MS", "400"))
AV1_SELFTEST_FILE = os.getenv("AV1_SELFTEST_FILE")  # optional file to decode at startup
NEGOTIATED_DCS = os.getenv("NEGOTIATED_DCS", "1") == "1"
DC_RESULTS_ID = int(os.getenv("DC_RESULTS_ID", "0"))
DC_CTRL_ID = int(os.getenv("DC_CTRL_ID", "1"))
SEND_GREETING = os.getenv("SEND_GREETING", "0") == "1"
DC_FACE_ID = int(os.getenv("DC_FACE_ID", "2"))
WAIT_FOR_ICE_MS = int(os.getenv("WAIT_FOR_ICE_MS", "0"))  # 0 = don't wait

def _noop(*_args, **_kwargs):
    return None

def _log_print(*args, **kwargs):
    if PRINT_LOGS:
        print(*args, **kwargs)

# ─────────────── Small logging helpers (print-only) ───────────────
def _ts():
    return time.strftime("%H:%M:%S")

def _ginfo(msg: str):
    _log_print(f"Srv 0 {_ts()} INFO: {msg}", flush=True)

def _gwarn(msg: str):
    _log_print(f"Srv 0 {_ts()} WARN: {msg}", flush=True)

def _gdebug(msg: str):
    _log_print(f"Srv 0 {_ts()} DEBUG: {msg}", flush=True)

def _exc_str(e: BaseException) -> str:
    """repr + traceback for returning in JSON and printing."""
    tb = traceback.format_exc()
    return f"{e!r}\n{tb}"

# Controla qué se exporta con: from connection.config import *
# Incluye también los nombres con guion bajo (privados) a propósito.
__all__ = [
    # Vars de entorno / config
    "PRINT_LOGS",
    "STUN_URL",
    "TURN_URL",
    "TURN_USER",
    "TURN_PASS",
    "POSE_USE_VIDEO",
    "ABSOLUTE_INTERVAL_MS",
    "IDLE_TO_FORCE_KF_MS",
    "FRAME_GAP_WARN_MS",
    "RESULTS_REQUIRE_ACK",
    "ACK_WARN_MS",
    "AV1_SELFTEST_FILE",
    "NEGOTIATED_DCS",
    "DC_RESULTS_ID",
    "DC_CTRL_ID",
    "SEND_GREETING",
    "DC_FACE_ID",
    "WAIT_FOR_ICE_MS",
    # Helpers
    "_env_bool",
    "_log_print",
    "_ts",
    "_ginfo",
    "_gwarn",
    "_gdebug",
    "_exc_str",
    "_noop",
]
