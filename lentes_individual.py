# main.py

import os
import requests
from pathlib import Path

# Importar las funciones clave del módulo detector
from deteccion_lentes_v2 import (
    configurar_optimizaciones_gpu,
    warm_up_modelo,
    get_glasses_probability,
    limpiar_cache_imagenes,
    obtener_estadisticas_cache
)

def descargar_imagen(url: str, nombre_archivo: str) -> str | None:
    """Descarga una imagen desde una URL y la guarda localmente."""
    try:
        print(f"[INFO] Descargando imagen de prueba desde: {url}")
        respuesta = requests.get(url, stream=True)
        respuesta.raise_for_status()  # Lanza un error si la descarga falla
        
        ruta_guardado = Path(nombre_archivo)
        with open(ruta_guardado, 'wb') as f:
            for chunk in respuesta.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"[INFO] ✅ Imagen guardada como: {ruta_guardado}")
        return str(ruta_guardado)
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] No se pudo descargar la imagen: {e}")
        return None

# --- Flujo Principal ---
if __name__ == "__main__":
    # 1. Configurar el detector. Cambia a `False` si prefieres usar CPU.
    # Si dlib no tiene soporte CUDA, cambiará a CPU automáticamente.
    configurar_optimizaciones_gpu(use_gpu=True)

    # 2. (Opcional pero recomendado) Calentar el modelo para inferencia rápida.
    warm_up_modelo()

    # 3. Descargar una imagen de ejemplo de internet
    # URL de una imagen de una persona con lentes
    url_imagen = "https://images.pexels.com/photos/1844547/pexels-photo-1844547.jpeg"
    ruta_local_imagen = "imagen_de_prueba.jpg"
    
    ruta_descargada = descargar_imagen(url_imagen, ruta_local_imagen)

    if ruta_descargada:
        # 4. Obtener la probabilidad de que la persona en la imagen use lentes
        print("\n[INFO] Analizando la imagen...")
        resultado = get_glasses_probability(ruta_descargada)

        # 5. Mostrar el resultado
        if isinstance(resultado, float):
            probabilidad_porcentaje = resultado * 100
            print("\n─────────── RESULTADO ───────────")
            print(f"🔬 Probabilidad de tener lentes: {probabilidad_porcentaje:.2f}%")
            if probabilidad_porcentaje > 30: # Umbral de ejemplo
                 print("👓 Es muy probable que la persona esté usando lentes.")
            else:
                 print("🙂 Es poco probable que la persona esté usando lentes.")
            print("─────────────────────────────────")
        else:
            # Manejar errores como 'No face detected'
            print(f"\n[AVISO] No se pudo procesar la imagen: {resultado}")
        
        # 6. Ver estadísticas del caché
        print("\n")
        obtener_estadisticas_cache()

        # 7. Limpieza: eliminar la imagen descargada
        print(f"\n[INFO] Limpiando archivo de prueba...")
        os.remove(ruta_descargada)
        
        # 8. Limpiar los cachés del detector
        limpiar_cache_imagenes()