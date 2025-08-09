# file: tests/face_recognition.py
import face_recognition
import numpy as np

C_intAreaFoto = 1080*2408

Altura_100_porciento = 2408

def FUN_CALCULAR_PORCENTAJE_ROSTRO(pImagen:np.ndarray):
    """
    Calcula el porcentaje de rostro de una imagen

    :parametros pImagen: np.ndarray de Numpy, obtenido de una imagen.
    :return: El porcentaje de rostro de la imagen
    """
    lLocacionesDeRostros = face_recognition.face_locations(pImagen)
    top, right, bottom, left = lLocacionesDeRostros[0]

    lAreaRostro = (right - left) * (bottom - top)

    lPorcentajeRostro = (lAreaRostro / C_intAreaFoto) * 100
    return lPorcentajeRostro

if __name__ == "__main__":
    ruta_imagen = r"C:\Users\Administrador\Downloads\Screenshot_20250808_203141_Gallery.jpg"
    img = face_recognition.load_image_file(ruta_imagen)  # returns RGB ndarray
    porcentaje = FUN_CALCULAR_PORCENTAJE_ROSTRO(img)
    print(f"El rostro ocupa aproximadamente {porcentaje:.2f}% de la imagen.")
