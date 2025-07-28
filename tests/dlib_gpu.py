import dlib

def verificar_compatibilidad_gpu_dlib():
    """
    Verifica e informa si dlib está compilado con soporte CUDA y si puede detectar una GPU.
    """
    print("--- Verificación de Compatibilidad de dlib con GPU (CUDA) ---")

    # dlib.DLIB_USE_CUDA es un valor booleano que es True si dlib fue compilado con soporte para CUDA.
    compilado_con_cuda = dlib.DLIB_USE_CUDA

    if not compilado_con_cuda:
        print("❌ **Fallo:** Tu versión de dlib fue instalada SIN soporte para CUDA.")
        print("   Para usar la GPU, necesitas instalar dlib desde el código fuente teniendo el CUDA Toolkit de NVIDIA preinstalado.")
        return

    print("✅ **Éxito:** Tu dlib fue compilado correctamente con soporte para CUDA.")

    # dlib.get_num_devices() devuelve el número de GPUs NVIDIA que dlib puede usar.
    # Será 0 si no se encuentra una GPU compatible.
    try:
        num_gpus_detectadas = dlib.get_num_devices()
    except Exception as e:
        print(f"\n❌ **Error Crítico:** Se produjo un error al intentar acceder a los dispositivos CUDA: {e}")
        print("   Esto usualmente indica una grave incompatibilidad entre el driver de NVIDIA, el CUDA Toolkit y la versión de dlib.")
        return

    print("\n--- Detección de Dispositivos Físicos ---")
    if num_gpus_detectadas > 0:
        print(f"✅ **¡Excelente!** dlib detectó {num_gpus_detectadas} GPU(s) disponibles.")
        print("   Tu configuración es correcta para el procesamiento acelerado por hardware.")
    else:
        print("❌ **Problema:** Aunque dlib tiene soporte CUDA, no pudo detectar ninguna GPU compatible en tu sistema.")
        print("   **Posibles Causas:**")
        print("   1. Los drivers de NVIDIA no están instalados o no funcionan correctamente.")
        print("   2. La versión de CUDA para la que se compiló dlib no es compatible con los drivers de tu sistema.")
        print("   3. No hay una GPU NVIDIA instalada en tu máquina.")

if __name__ == "__main__":
    verificar_compatibilidad_gpu_dlib()