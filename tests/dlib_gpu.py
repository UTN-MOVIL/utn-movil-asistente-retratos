#!/usr/bin/env python3

# check_dlib_gpu.py
# A script to verify full dlib GPU (CUDA) compatibility.

import sys
import traceback  # Import the traceback module
from pathlib import Path

import numpy as np

# --- Configuration ---
# The script expects the model to be in a 'models' subfolder.
# Download from: http://dlib.net/files/mmod_human_face_detector.dat.bz2
try:
    # To this:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

MODELS_DIR = BASE_DIR / "models"
CNN_FACE_DETECTOR_PATH = MODELS_DIR / "mmod_human_face_detector.dat"

# --- Main Verification Logic ---
def check_dlib_gpu_compatibility():
    """
    Performs a series of checks to verify full dlib GPU compatibility.
    """
    print("--- Verificador de Compatibilidad de dlib con GPU ---")

    # This is a critical first step. If dlib is not available, nothing else matters.
    try:
        import dlib
    except ImportError:
        print("\n❌ FALLO CRÍTICO: La librería 'dlib' no está instalada.")
        print("    -> Solución: Instala dlib usando 'pip install dlib' o compílala desde la fuente.")
        sys.exit(1)

    # 1. Check if dlib was compiled with CUDA support
    print("\n[Paso 1/4] Verificando la compilación de dlib con CUDA...")
    if dlib.DLIB_USE_CUDA:
        print("    ✅ ÉXITO: Tu instalación de dlib fue compilada con soporte para CUDA.")
    else:
        print("\n❌ FALLO: Tu instalación de dlib NO fue compilada con soporte para CUDA.")
        print("    -> Solución: Reinstala dlib desde la fuente, asegurándote de que CMake encuentre tu kit de herramientas CUDA.")
        sys.exit(1)

    # 2. Check if dlib can detect CUDA-enabled devices
    print("\n[Paso 2/4] Detectando dispositivos GPU disponibles...")
    try:
        num_devices = dlib.cuda.get_num_devices()
        if num_devices > 0:
            print(f"    ✅ ÉXITO: dlib detectó {num_devices} dispositivo(s) GPU compatibles con CUDA.")
        else:
            print("\n❌ FALLO: dlib no detectó ningún dispositivo GPU compatible con CUDA.")
            print("    -> Solución: Asegúrate de que los drivers de NVIDIA estén instalados y actualizados.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ FALLO: Ocurrió un error al detectar dispositivos GPU: {e}")
        print("    -> Solución: Puede haber un problema con la instalación de tus drivers de NVIDIA o el kit de herramientas CUDA.")
        sys.exit(1)

    # 3. Check if the required CNN model file exists
    print("\n[Paso 3/4] Verificando la existencia del modelo CNN facial...")
    if not CNN_FACE_DETECTOR_PATH.exists():
        print(f"\n❌ FALLO: No se encontró el archivo del modelo en: {CNN_FACE_DETECTOR_PATH}")
        print(f"    -> Solución: Descarga 'mmod_human_face_detector.dat.bz2', descomprímelo y colócalo en la carpeta '{MODELS_DIR}'.")
        print("      Link de descarga: http://dlib.net/files/mmod_human_face_detector.dat.bz2")
        # Create the directory if it doesn't exist to help the user
        if not MODELS_DIR.exists():
            print(f"    [INFO] Creando el directorio '{MODELS_DIR}' para ti.")
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
        sys.exit(1)
    else:
        print(f"    ✅ ÉXITO: Modelo CNN encontrado en '{CNN_FACE_DETECTOR_PATH}'.")

    # 4. Attempt to load the model and perform a test inference
    print("\n[Paso 4/4] Cargando el modelo en la GPU y realizando una prueba de inferencia...")
    try:
        detector = dlib.cnn_face_detection_model_v1(str(CNN_FACE_DETECTOR_PATH))
        print("    - Modelo CNN cargado exitosamente en la memoria.")

        # Create a dummy image for a warm-up inference
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Run the detector. This loads the model onto the GPU VRAM and tests it.
        detector(dummy_image, 1)
        print("    - Inferencia de prueba en la GPU completada con éxito.")

    except RuntimeError as e:
        print(f"\n❌ FALLO: Ocurrió un error en tiempo de ejecución al usar el modelo en la GPU.")
        print("\n" + "="*20 + " ANÁLISIS COMPLETO DEL ERROR " + "="*20)
        
        print("\n--- Detalles Principales ---")
        print(f"   Tipo de Error: {type(e).__name__}")
        print(f"   Módulo del Error: {type(e).__module__}")
        print(f"   Argumentos del Error: {e.args}")
        print(f"   Mensaje Principal: {str(e)}")
        print(f"   Representación Completa: {repr(e)}")
        
        # Print error attributes if they exist
        print("\n--- Atributos del Error ---")
        error_attrs = [attr for attr in dir(e) if not attr.startswith('_')]
        if error_attrs:
            for attr in error_attrs:
                try:
                    value = getattr(e, attr)
                    if not callable(value):  # Skip methods
                        print(f"   {attr}: {value}")
                except:
                    print(f"   {attr}: <no se pudo acceder>")
        else:
            print("   No hay atributos adicionales disponibles.")
        
        # Print the full traceback with more details
        print("\n--- Traceback Completo del Error ---")
        import traceback
        tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
        for line in tb_lines:
            print(f"   {line.rstrip()}")
        
        # Print local variables at the point of error (if traceback exists)
        print("\n--- Variables Locales en el Punto del Error ---")
        if e.__traceback__:
            tb = e.__traceback__
            while tb.tb_next:  # Go to the innermost frame
                tb = tb.tb_next
            
            local_vars = tb.tb_frame.f_locals
            if local_vars:
                for var_name, var_value in local_vars.items():
                    try:
                        # Avoid printing very large objects
                        str_value = str(var_value)
                        if len(str_value) > 200:
                            str_value = str_value[:200] + "... (truncado)"
                        print(f"   {var_name}: {str_value}")
                    except:
                        print(f"   {var_name}: <no se pudo convertir a string>")
            else:
                print("   No hay variables locales disponibles.")
        
        # Additional GPU/CUDA specific diagnostics
        print("\n--- Diagnósticos Adicionales de GPU/CUDA ---")
        try:
            print(f"   Número de dispositivos CUDA detectados: {dlib.cuda.get_num_devices()}")
        except:
            print("   No se pudo obtener información de dispositivos CUDA")
        
        try:
            print(f"   dlib.DLIB_USE_CUDA: {dlib.DLIB_USE_CUDA}")
        except:
            print("   No se pudo verificar el estado de CUDA en dlib")
        
        # Try to get CUDA/GPU memory info if available
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"   Memoria GPU (nvidia-smi): {result.stdout.strip()}")
        except:
            print("   No se pudo obtener información de memoria GPU")
        
        print("\n" + "="*72)
        
        # Optional: Save error details to a file
        try:
            error_log_path = Path("dlib_error_log.txt")
            with open(error_log_path, "w", encoding="utf-8") as f:
                f.write("DLIB GPU ERROR LOG\n")
                f.write("="*50 + "\n")
                f.write(f"Error Type: {type(e).__name__}\n")
                f.write(f"Error Message: {str(e)}\n")
                f.write(f"Error Args: {e.args}\n\n")
                f.write("Full Traceback:\n")
                f.write("".join(traceback.format_exception(type(e), e, e.__traceback__)))
                
            print(f"   📝 Detalles del error guardados en: {error_log_path}")
        except:
            print("   ⚠️  No se pudo guardar el log de error en archivo")
        
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ FALLO: Ocurrió un error inesperado: {e}")
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)

    print("\n" + "="*60)
    print("🎉 ¡FELICITACIONES! Tu entorno está correctamente configurado para usar dlib con aceleración por GPU.")
    print("="*60)

if __name__ == "__main__":
    check_dlib_gpu_compatibility()