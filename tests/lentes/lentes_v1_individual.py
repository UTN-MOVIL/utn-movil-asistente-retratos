# main_local.py

import os
import time  # Importado para medir el tiempo de ejecución

# Asegúrate de que tu archivo 'deteccion_lentes_v2.py' esté en el mismo directorio
# o sea accesible a través del PYTHONPATH.
# Importar las funciones clave del módulo detector
from modulos import (
    configurar_optimizaciones_gpu,
    warm_up_modelo,
    get_glasses_probability,
    obtener_estadisticas_cache
)

# --- Flujo Principal ---
if __name__ == "__main__":
    # 1. Configurar el detector. Cambia a `False` si prefieres usar CPU.
    # Si dlib no tiene soporte CUDA, cambiará a CPU automáticamente.
    configurar_optimizaciones_gpu()

    # 2. (Opcional pero recomendado) Calentar el modelo para inferencia rápida.
    warm_up_modelo()

    # 3. Especificar la ruta a tu imagen local
    # ⬇️⬇️⬇️ CAMBIA ESTA LÍNEA POR LA RUTA DE TU IMAGEN ⬇️⬇️⬇️
    ruta_imagen_local = r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE\TESIS\CODIGO\funcionalidades_validador_retratos\results\image_cache\0401775143.jpg"
    # Ejemplos:
    # - En Windows: "C:\\Users\\TuUsuario\\Fotos\\mi_foto.png"
    # - En macOS/Linux: "/home/usuario/imagenes/selfie.jpg"

    # 4. Verificar si el archivo de imagen existe antes de continuar
    if os.path.exists(ruta_imagen_local):
        # 5. Obtener la probabilidad de que la persona en la imagen use lentes
        print(f"\n[INFO] Analizando la imagen local: {ruta_imagen_local}")
        
        # Iniciar cronómetro antes de la inferencia
        start_time = time.time()
        resultado = get_glasses_probability(ruta_imagen_local)
        # Detener cronómetro y calcular la duración
        inference_time = time.time() - start_time

        # 6. Mostrar el resultado
        if isinstance(resultado, float):
            probabilidad_porcentaje = resultado * 100
            print("\n─────────── RESULTADO ───────────")
            print(f"🔬 Probabilidad de tener lentes: {probabilidad_porcentaje:.2f}%")
            
            # Mostrar el tiempo de inferencia calculado
            print(f"⏱️ Tiempo de inferencia: {inference_time:.4f} segundos.")
            
            # Puedes ajustar este umbral según la precisión que observes
            if probabilidad_porcentaje > 30:
                print("👓 Es muy probable que la persona esté usando lentes.")
            else:
                print("🙂 Es poco probable que la persona esté usando lentes.")
            print("─────────────────────────────────")
        else:
            # Manejar errores como 'No face detected' o si el archivo no es una imagen válida
            print(f"\n[AVISO] No se pudo procesar la imagen: {resultado}")
            # Mostrar el tiempo incluso si hubo un error en el análisis
            print(f"⏱️ Tiempo de ejecución: {inference_time:.4f} segundos.")
        
        # 7. Ver estadísticas del caché (opcional)
        print("\n")
        obtener_estadisticas_cache()
        print("\n[INFO] Proceso finalizado.")

    else:
        # Mensaje de error si la ruta especificada no existe
        print(f"[ERROR] No se encontró ninguna imagen en la ruta especificada: {ruta_imagen_local}")
        print("[INFO] Por favor, verifica que la ruta sea correcta y que el archivo exista.")