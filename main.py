#!/usr/bin/env python3
import os
import io
import sys
import tempfile
from tqdm import tqdm
from typing import List, Tuple

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# CAMBIO 1: Importar funciones optimizadas en lugar de la versión básica
from deteccion_lentes_v1 import (
    get_glasses_probability,
    get_glasses_probability_batch,
    configurar_optimizaciones_gpu,
    warm_up_modelo,
    obtener_estadisticas_cache,
    limpiar_cache_imagenes
)

from exportacion_datos_excel import (
    format_to_hyperlinks,
    normalize_dict_lengths,
    dict_to_excel,
    get_file_count,
)

# ── Google Drive ──────────────────────────────────────────────────────────────
SCOPES     = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_FILE = "token.json"
CREDS_FILE = "credentials.json"

def drive_service():
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    else:
        flow  = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return build("drive", "v3", credentials=creds)

def get_folder_id_by_path(path: str, drive):
    """Del estilo '/Mi unidad/Carp1/Carp2' devuelve el ID de la última carpeta."""
    segments  = [s for s in path.strip("/").split("/") if s and s != "Mi unidad"]
    parent_id = "root"
    for name in segments:
        resp = (
            drive.files()
            .list(
                q=(
                    f"name = '{name}' and "
                    "mimeType = 'application/vnd.google-apps.folder' and "
                    f"'{parent_id}' in parents and trashed = false"
                ),
                fields="files(id)",
                pageSize=1,
            )
            .execute()
        )
        items = resp.get("files", [])
        if not items:
            raise FileNotFoundError(f"Carpeta '{name}' no encontrada (parent={parent_id})")
        parent_id = items[0]["id"]
    return parent_id

def list_files_recursive(folder_id: str, drive) -> List[Tuple[str, str]]:
    """
    Devuelve pares (file_id, drive_path) de todos los archivos (no carpetas)
    dentro de la carpeta indicada y sus subcarpetas.
    """
    results = []

    # primero listamos el contenido directo
    query = f"'{folder_id}' in parents and trashed = false"
    page_token = None
    while True:
        resp = (
            drive.files()
            .list(
                q=query,
                fields=(
                    "nextPageToken, "
                    "files(id, name, mimeType, parents)"
                ),
                pageToken=page_token,
            )
            .execute()
        )
        for f in resp["files"]:
            if f["mimeType"] == "application/vnd.google-apps.folder":
                # recursión en subcarpeta
                results.extend(list_files_recursive(f["id"], drive))
            else:
                results.append((f["id"], f["name"]))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results

def download_file(file_id: str, dest_path: str, drive):
    """Descarga un archivo de Drive al path local indicado."""
    request = drive.files().get_media(fileId=file_id)
    fh      = io.FileIO(dest_path, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()

# ── NUEVA VERSIÓN OPTIMIZADA: Procesamiento con batch ───────────────────────
def process_drive_folder_optimized(drive_folder_path: str, usar_batch: bool = True, 
                                 umbral_minimo: float = 0.0) -> Tuple[List[str], List[float]]:
    """
    VERSIÓN ULTRA-OPTIMIZADA del procesamiento de carpeta Drive.
    
    Args:
        drive_folder_path: Ruta de la carpeta en Drive
        usar_batch: Si True, usa procesamiento en batch (MUY recomendado)
        umbral_minimo: Umbral mínimo de confianza para filtrar detecciones
    
    Returns:
        Tupla con (rutas_locales, probabilidades_lentes)
    """
    print("[INFO] 🚀 Iniciando procesamiento ultra-optimizado...")
    
    # Configurar optimizaciones al inicio
    configurar_optimizaciones_gpu()
    warm_up_modelo()
    
    drive = drive_service()
    folder_id = get_folder_id_by_path(drive_folder_path, drive)

    files = list_files_recursive(folder_id, drive)
    if not files:
        print("No se encontraron archivos.")
        return [], []

    print(f"[INFO] Encontrados {len(files)} archivos para procesar")
    
    temp_dir = tempfile.mkdtemp(prefix="glasses_optimized_")
    print(f"[INFO] Directorio temporal: {temp_dir}")
    
    # FASE 1: Descarga de archivos (con filtrado por extensión)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths: List[str] = []
    
    print("[INFO] 📥 Descargando archivos...")
    for file_id, name in tqdm(files, desc="Descargando", unit="archivo"):
        # Filtrar por extensión antes de descargar
        if not any(name.lower().endswith(ext) for ext in valid_extensions):
            continue
            
        local_path = os.path.join(temp_dir, name)
        try:
            download_file(file_id, local_path, drive)
            image_paths.append(local_path)
        except Exception as e:
            print(f"[ERROR] Saltando descarga de {name!r}: {e}")
            continue
    
    if not image_paths:
        print("[WARNING] No se descargaron imágenes válidas")
        return [], []
    
    print(f"[INFO] ✅ Descargadas {len(image_paths)} imágenes")
    
    # FASE 2: Procesamiento optimizado de detección de lentes
    print("[INFO] 🔍 Iniciando detección de lentes...")
    
    if usar_batch and len(image_paths) > 1:
        # PROCESAMIENTO EN BATCH (ULTRA-RÁPIDO)
        print(f"[INFO] Usando procesamiento en batch para {len(image_paths)} imágenes")
        glasses_probs = get_glasses_probability_batch(image_paths, umbral_minimo)
        
        # Mostrar progreso y estadísticas
        detecciones_positivas = sum(1 for p in glasses_probs if p > 0.5)
        print(f"[INFO] ✅ Procesamiento batch completado")
        print(f"[INFO] 📊 Detecciones positivas: {detecciones_positivas}/{len(glasses_probs)}")
        
    else:
        # PROCESAMIENTO INDIVIDUAL (para casos especiales)
        print("[INFO] Usando procesamiento individual optimizado")
        glasses_probs: List[float] = []
        
        for path in tqdm(image_paths, desc="Detectando lentes", unit="imagen"):
            try:
                prob = get_glasses_probability(path, umbral_minimo)
                glasses_probs.append(prob)
            except Exception as e:
                print(f"[ERROR] Error procesando {path}: {e}")
                glasses_probs.append(0.0)
    
    # FASE 3: Estadísticas finales
    print("[INFO] 📈 Estadísticas finales:")
    obtener_estadisticas_cache()
    
    # Estadísticas de detección
    total_imagenes = len(glasses_probs)
    con_lentes = sum(1 for p in glasses_probs if p >= 0.5)
    sin_lentes = total_imagenes - con_lentes
    prob_promedio = sum(glasses_probs) / total_imagenes if total_imagenes > 0 else 0
    
    print(f"[INFO] 👓 Con lentes: {con_lentes} ({con_lentes/total_imagenes*100:.1f}%)")
    print(f"[INFO] 👁️  Sin lentes: {sin_lentes} ({sin_lentes/total_imagenes*100:.1f}%)")
    print(f"[INFO] 📊 Probabilidad promedio: {prob_promedio:.3f}")
    
    return image_paths, glasses_probs

# ── VERSIÓN COMPATIBLE (mantiene la interfaz original) ───────────────────────
def process_drive_folder(drive_folder_path: str) -> Tuple[List[str], List[float]]:
    """
    Versión compatible que usa internamente las optimizaciones.
    Mantiene la misma interfaz que el código original.
    """
    return process_drive_folder_optimized(drive_folder_path, usar_batch=True)

# ── Main con opciones avanzadas ──────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 DETECTOR DE LENTES ULTRA-OPTIMIZADO")
    print("=" * 50)
    
    # Configuración
    dataset_drive_path = (
        "/Mi unidad/INGENIERIA_EN_SOFTWARE/6to_Semestre/"
        "PRACTICAS/Practicas-FOTOS/Primera_Revision/"
        "validator/results/validated_color"
    )
    
    # OPCIONES DE CONFIGURACIÓN
    USAR_BATCH = True           # True para máximo rendimiento
    UMBRAL_MINIMO = 0.0        # Umbral mínimo de confianza
    UMBRAL_DETECCION = 0.5     # Umbral para considerar "con lentes"
    
    results_folder = "results"
    os.makedirs(results_folder, exist_ok=True)

    try:
        # Usar versión optimizada
        print(f"[INFO] Procesando carpeta: {dataset_drive_path}")
        paths, probs = process_drive_folder_optimized(
            dataset_drive_path, 
            usar_batch=USAR_BATCH,
            umbral_minimo=UMBRAL_MINIMO
        )
        
        if not paths:
            print("[ERROR] No se procesaron imágenes. Abortando.")
            sys.exit(1)
        
        # Preparar datos para Excel con estadísticas adicionales
        informacion = {
            "Rutas": format_to_hyperlinks(paths),
            "Probabilidad de tener lentes": probs,
            "Detección (≥0.5)": ["SÍ" if p >= UMBRAL_DETECCION else "NO" for p in probs],
            "Confianza": ["Alta" if p >= 0.8 else "Media" if p >= 0.5 else "Baja" for p in probs]
        }

        normalized = normalize_dict_lengths(informacion)
        output_file = dict_to_excel(
            normalized,
            f"{results_folder}/Reporte_probabilidad_lentes_OPTIMIZADO_{get_file_count(results_folder)+1}.xlsx",
        )
        
        print(f"\n✅ PROCESAMIENTO COMPLETADO")
        print(f"📊 Excel generado en: {output_file}")
        print(f"📁 Archivos procesados: {len(paths)}")
        
        # Limpiar caché al final (opcional)
        # limpiar_cache_imagenes()
        
    except KeyboardInterrupt:
        print("\n[INFO] Proceso interrumpido por el usuario")
        limpiar_cache_imagenes()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Error durante el procesamiento: {e}")
        limpiar_cache_imagenes()
        sys.exit(1)