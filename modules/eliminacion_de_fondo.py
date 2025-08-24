import cv2
import numpy as np
from rembg import remove
from rembg.session_factory import new_session

def FUN_OBTENER_IMAGEN_SIN_FONDO(pImagenOriginal: np.ndarray) -> np.ndarray:
    """
    Obtiene una imagen sin fondo a partir de una imagen original utilizando la librería rembg.

    :param pImagenOriginal: Imagen original en formato numpy array.
    :return: Imagen resultante sin fondo en formato numpy array (con canal alfa) o None en caso de error.
    """
    try:
        # Codificar la imagen original a bytes en formato PNG
        _, datos_img_codificados = cv2.imencode('.png', pImagenOriginal)
        bytes_datos_img = datos_img_codificados.tobytes()

        # Crear una sesión utilizando el modelo u2net_human_seg
        session = new_session("u2net_human_seg")

        # Eliminar el fondo de la imagen
        output_bytes = remove(bytes_datos_img, session=session)

        # Convertir los bytes resultantes a un array de numpy
        arreglo_img = np.frombuffer(output_bytes, dtype=np.uint8)
        img_decodificada = cv2.imdecode(arreglo_img, cv2.IMREAD_UNCHANGED)

        return img_decodificada

    except Exception as e:
        print("Error al procesar la imagen:", e)
        return None
    
if __name__ == "__main__":
    # Cargar imagen desde archivo
    imagen_entrada = cv2.imread(r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE\PROYECTO_FOTOGRAFIAS_ESTUDIANTES\datasets\validated_color\1005072473.jpg")

    # Verifica que la imagen se haya cargado correctamente
    if imagen_entrada is None:
        print("No se pudo cargar la imagen.")
    else:
        # Procesar imagen para eliminar el fondo
        imagen_sin_fondo = FUN_OBTENER_IMAGEN_SIN_FONDO(imagen_entrada)

        # Guardar la imagen resultante
        if imagen_sin_fondo is not None:
            cv2.imwrite("imagen_sin_fondo.png", imagen_sin_fondo)
            print("Imagen sin fondo guardada como 'imagen_sin_fondo.png'")
        else:
            print("No se pudo procesar la imagen.")