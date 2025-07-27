#!/usr/bin/env python3
"""
Detección de lentes en imágenes usando Autodistill + Detic (OPTIMIZADO).
Requisitos:
    pip install autodistill-detic supervision opencv-python torch torchvision tqdm
    (y clonar el repo oficial de Detic en una sub-carpeta ./Detic)
"""
import os, sys, urllib.request, shutil, tqdm, torch, supervision as sv, cv2
import numpy as np
from pathlib import Path
from autodistill_detic import DETIC
from autodistill.detection import CaptionOntology

# ───────────────── 1. Pesos ──────────────────────────────────────────────────
WEIGHTS_NAME = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
WEIGHTS_URL  = f"https://dl.fbaipublicfiles.com/detic/{WEIGHTS_NAME}"
CACHE_DIR    = Path.home()/".cache"/"autodistill"/"Detic"/"models"
WEIGHTS_LOCAL= CACHE_DIR/WEIGHTS_NAME

if not WEIGHTS_LOCAL.exists():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Descargando {WEIGHTS_NAME} …")
    with urllib.request.urlopen(WEIGHTS_URL) as r, open(WEIGHTS_LOCAL,"wb") as f:
        total=int(r.headers["Content-Length"])
        with tqdm.tqdm(total=total,unit="B",unit_scale=True) as p:
            while (chunk:=r.read(8192)):
                f.write(chunk); p.update(len(chunk))
    print(f"[INFO] ✅  Pesos en {WEIGHTS_LOCAL}")
else:
    print(f"[INFO] ✅  Pesos ya presentes en {WEIGHTS_LOCAL}")

# ───────────────── 2. Rutas Detic ────────────────────────────────────────────
detic_root = Path(__file__).parent/"Detic"          # repo clonado
cfg_file   = detic_root/"configs"/WEIGHTS_NAME.replace(".pth",".yaml")
print(f"[INFO] Detic root            : {detic_root} | existe → {detic_root.is_dir()}")
print(f"[INFO] YAML original         : {cfg_file}  | existe → {cfg_file.exists()}")

# Añade Detic al PYTHONPATH
if str(detic_root) not in sys.path:
    sys.path.insert(0, str(detic_root))
    print(f"[INFO] sys.path ← {detic_root}")

# ─────────── 3. Copia YAML al directorio de trabajo ./configs ────────────────
local_cfg_dir = Path.cwd()/"configs"
local_cfg_dir.mkdir(exist_ok=True)
target_yaml   = local_cfg_dir/cfg_file.name
if not target_yaml.exists():
    shutil.copy(cfg_file, target_yaml)
    print(f"[INFO] Copiado YAML → {target_yaml}")
else:
    print(f"[INFO] YAML ya presente en {target_yaml}")

# (¡NO cambiamos de directorio! El cwd sigue siendo {Path.cwd()})
print(f"[INFO] Directorio de trabajo actual → {Path.cwd()}")

# ─────────── 4. Construir modelo Detic ───────────────────────────────────────
ontology    = CaptionOntology({"eyeglasses":"eyeglasses"})
print("[INFO] Creando modelo DETIC …")
detic_model = DETIC(ontology=ontology)              # ahora sí inicia
device      = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Modelo listo en {device}")

# OPTIMIZACIÓN: Pre-cargar imagen en memoria para evitar I/O repetido
_image_cache = {}
MAX_CACHE_SIZE = 50  # Limitar tamaño de caché

def _load_image_optimized(ruta_imagen: str) -> np.ndarray:
    """Carga imagen con caché en memoria y redimensionado opcional."""
    global _image_cache
    
    # Limpiar caché si está muy lleno
    if len(_image_cache) > MAX_CACHE_SIZE:
        _image_cache.clear()
    
    if ruta_imagen not in _image_cache:
        if not os.path.exists(ruta_imagen):
            raise FileNotFoundError(f"Imagen no encontrada: {ruta_imagen}")
            
        img = cv2.imread(ruta_imagen)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
            
        # OPTIMIZACIÓN: Redimensionar imagen si es muy grande (>1024px en cualquier dimensión)
        h, w = img.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"[INFO] Redimensionada {ruta_imagen}: {w}x{h} → {new_w}x{new_h}")
        
        _image_cache[ruta_imagen] = img
    
    return _image_cache[ruta_imagen]

# ─────────── 5. Utilidades de inferencia OPTIMIZADAS ────────────────────────
def get_glasses_probability(ruta_imagen: str, umbral_min: float = 0.0) -> float:
    """
    Versión optimizada que:
    1. Usa caché de imágenes en memoria
    2. Redimensiona imágenes grandes
    3. Evita operaciones innecesarias
    4. Retorna directamente el máximo sin crear listas intermedias
    """
    try:
        # Usar imagen cacheada/optimizada
        img = _load_image_optimized(ruta_imagen)
        
        # OPTIMIZACIÓN: Pasar imagen como array numpy directamente
        dets: sv.Detections = detic_model.predict(img)
        
        # OPTIMIZACIÓN: Calcular máximo directamente sin crear lista intermedia
        if dets.confidence is None or len(dets.confidence) == 0:
            return 0.0
            
        max_conf = 0.0
        for conf in dets.confidence:
            if conf >= umbral_min and conf > max_conf:
                max_conf = conf
                
        return max_conf
        
    except Exception as e:
        print(f"[ERROR] Error procesando {ruta_imagen}: {e}")
        return 0.0

def verificar_presencia_de_lentes(ruta_imagen: str, umbral: float = 0.5) -> str:
    """Función principal con logging mejorado."""
    prob = get_glasses_probability(ruta_imagen, umbral)
    msg = (f"Imagen contiene lentes (prob.≈{prob:.2f})"
           if prob >= umbral else
           f"Imagen NO contiene lentes (prob.≈{prob:.2f})")
    print(f"[INFO] {msg}")
    return msg

# ─────────── 6. Función para procesar múltiples imágenes en lote ─────────────
def procesar_lote_imagenes(rutas_imagenes: list, umbral: float = 0.5, 
                          mostrar_progreso: bool = True) -> dict:
    """
    Procesa múltiples imágenes de forma optimizada.
    Retorna diccionario con rutas como keys y probabilidades como values.
    """
    resultados = {}
    
    if mostrar_progreso:
        rutas_imagenes = tqdm.tqdm(rutas_imagenes, desc="Procesando imágenes")
    
    for ruta in rutas_imagenes:
        try:
            prob = get_glasses_probability(ruta, 0.0)  # umbral mínimo 0 para obtener todas
            resultados[ruta] = prob
            if mostrar_progreso:
                status = "✓ LENTES" if prob >= umbral else "✗ sin lentes"
                rutas_imagenes.set_postfix_str(f"{status} ({prob:.2f})")
        except Exception as e:
            print(f"[ERROR] {ruta}: {e}")
            resultados[ruta] = 0.0
    
    return resultados

# ─────────── 7. Función para limpiar caché manualmente ───────────────────────
def limpiar_cache_imagenes():
    """Limpia el caché de imágenes para liberar memoria."""
    global _image_cache
    _image_cache.clear()
    print(f"[INFO] Caché de imágenes limpiado")