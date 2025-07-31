#!/usr/bin/env python3
import os
import cv2
import numpy as np

MAX_CACHE_SIZE = 100
_image_cache: dict[str, np.ndarray] = {}


def _get_image_hash(ruta_imagen: str) -> str:
    try:
        st = os.stat(ruta_imagen)
        return f"{ruta_imagen}_{st.st_mtime}_{st.st_size}"
    except FileNotFoundError:
        return ruta_imagen


def _load_image_optimized(ruta_imagen: str) -> np.ndarray:
    img_hash = _get_image_hash(ruta_imagen)

    if len(_image_cache) > MAX_CACHE_SIZE:
        for key in list(_image_cache)[: MAX_CACHE_SIZE // 5]:
            _image_cache.pop(key)

    if img_hash not in _image_cache:
        if not os.path.exists(ruta_imagen):
            raise FileNotFoundError(f"Imagen no encontrada: {ruta_imagen}")

        img = cv2.imread(ruta_imagen)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")

        h, w = img.shape[:2]
        target = 640
        if max(h, w) > target:
            scale = target / max(h, w)
            img = cv2.resize(
                img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
            )
            print(f"[INFO] Redimensionada {ruta_imagen} a ≤{target}px")

        _image_cache[img_hash] = img

    return _image_cache[img_hash]

print(_load_image_optimized("C:\\Users\\Administrador\\Documents\\INGENIERIA_EN_SOFTWARE\\TESIS\\CODIGO\\funcionalidades_validador_retratos\\results\\image_cache\\1003414107.jpg"))  # Ejemplo de uso
