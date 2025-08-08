import cv2
from libreria.validadores.postura_cuerpo import FUN_OBTENER_POSICIONES_HOMBROS
import math
import pandas as pd
import os
import face_recognition
import numpy as np
import time

def FUN_VALIDAR_POSTURA_ROSTRO(pImagen):
    lstrObservaciones = ""
    ldicPosicionesOjos = FUN_OBTENER_POSICIONES_OJOS(pImagen)
    ltupPosicionMenton = FUN_OBTENER_POSICION_MENTON(pImagen)
    llisPosicionesHombros = FUN_OBTENER_POSICIONES_HOMBROS(pImagen)
    lfloInclinacionOjos = -1*FUN_CALCULAR_INCLINACION(ldicPosicionesOjos.get('ojo_izquierdo', None), ldicPosicionesOjos.get('ojo_derecho', None))

    if len(llisPosicionesHombros) < 2:
        return {'valido': 'False'}, "No se encontraron ambos hombros"
    
    ltupPosicionHombroDerecho = llisPosicionesHombros[1]
    ltupPosicionHombroIzquierdo = llisPosicionesHombros[0]

    lfloAnguloHombroIzquierdoMenton, lfloAnguloHombroDerechoMenton = FUN_CALCULAR_ANGULOS(ltupPosicionHombroIzquierdo, ltupPosicionMenton, ltupPosicionHombroDerecho)
    lfloVariacionAnguloHombrosMenton = lfloAnguloHombroIzquierdoMenton - lfloAnguloHombroDerechoMenton

    lbooPosicionOjosValida = abs(lfloInclinacionOjos) <= 0.20
    lbooPosicionMentonValida = abs(lfloVariacionAnguloHombrosMenton) <= 15

    if not lbooPosicionOjosValida or not lbooPosicionMentonValida:
        lstrObservaciones += "Verifique que su rostro esté recto\n"
    
    return {'valido': str(lbooPosicionOjosValida)}, lstrObservaciones

def FUN_OBTENER_POSICIONES_OJOS(imagen_cv2):
    """
    Detecta y retorna las posiciones de los ojos en un rostro utilizando face_landmarks.
    Además, dibuja cada uno de los puntos de left_eye y right_eye sobre la imagen.

    Se asume que la imagen ingresada es una imagen en formato BGR (como la que retorna cv2.imread).

    Parámetros:
        imagen_cv2 (numpy.ndarray): Imagen en formato BGR.

    Retorna:
        tuple: Un diccionario con las posiciones de los ojos y la imagen modificada.
               Ejemplo:
               (
                  {'left_eye': [(x1, y1), (x2, y2), ...], 'right_eye': [(x1, y1), (x2, y2), ...]},
                  imagen_cv2 con los puntos dibujados
               )
    """
    # Convertir la imagen de BGR a RGB para el procesamiento
    imagen_rgb = cv2.cvtColor(imagen_cv2, cv2.COLOR_BGR2RGB)
    
    # Detectar los landmarks de los rostros en la imagen
    rostros_landmarks = face_recognition.face_landmarks(imagen_rgb)
    
    # Si no se detecta ningún rostro, retorna un diccionario vacío y la imagen original
    if not rostros_landmarks:
        return {}, imagen_cv2
    
    # Para este ejemplo, se toma el primer rostro detectado
    landmarks = rostros_landmarks[0]
    
    # Extraer las posiciones de los ojos
    left_eye = landmarks.get('left_eye', [])
    right_eye = landmarks.get('right_eye', [])
    
    # pts_left_eye = np.array(left_eye, np.int32)
    # pts_left_eye = pts_left_eye.reshape((-1, 1, 2))
    # cv2.polylines(imagen_cv2, [pts_left_eye], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # pts_right_eye = np.array(right_eye, np.int32)
    # pts_right_eye = pts_right_eye.reshape((-1, 1, 2))
    # cv2.polylines(imagen_cv2, [pts_right_eye], isClosed=True, color=(0, 255, 0), thickness=2)

    centroide_izquierdo = FUN_CALCULAR_CENTROIDE(left_eye)
    centroide_izquierdo = (int(centroide_izquierdo[0]), int(centroide_izquierdo[1]))
    # cv2.circle(imagen_cv2, centroide_izquierdo, radius=3, color=(0, 255, 0), thickness=-1)

    centroide_derecho = FUN_CALCULAR_CENTROIDE(right_eye)
    centroide_derecho = (int(centroide_derecho[0]), int(centroide_derecho[1]))
    # cv2.circle(imagen_cv2, centroide_derecho, radius=3, color=(0, 255, 0), thickness=-1)
    
    # cv2.imwrite("RESULTADO_POSTURA_ROSTRO.jpg", imagen_cv2)
    
    return {'ojo_izquierdo': centroide_izquierdo, 'ojo_derecho': centroide_derecho}

def FUN_CALCULAR_CENTROIDE(puntos):
    """
    Calcula el centroide de un polígono dado una lista de puntos.
    
    Parámetros:
        puntos (list of tuple): Lista de tuplas (x, y) que definen los vértices del polígono, 
                                ordenados de forma secuencial (puede ser en sentido horario o antihorario).
    
    Retorna:
        tuple: (cx, cy) que es el centroide del polígono.
        
    Nota:
        Se utiliza la fórmula del centroide de un polígono:
            A = 1/2 * Σ (x_i * y_(i+1) - x_(i+1) * y_i)
            Cx = 1/(6A) * Σ (x_i + x_(i+1)) * (x_i * y_(i+1) - x_(i+1) * y_i)
            Cy = 1/(6A) * Σ (y_i + y_(i+1)) * (x_i * y_(i+1) - x_(i+1) * y_i)
    """
    if len(puntos) < 3:
        raise ValueError("Se necesitan al menos tres puntos para formar un polígono.")

    area = 0
    cx = 0
    cy = 0
    n = len(puntos)
    
    for i in range(n):
        x_i, y_i = puntos[i]
        x_next, y_next = puntos[(i + 1) % n]  # % n para cerrar el polígono
        factor = x_i * y_next - x_next * y_i
        area += factor
        cx += (x_i + x_next) * factor
        cy += (y_i + y_next) * factor

    area *= 0.5
    if area == 0:
        raise ValueError("El área del polígono es cero. Es probable que los puntos sean colineales.")
    
    cx /= (6 * area)
    cy /= (6 * area)
    
    return (cx, cy)

def FUN_CALCULAR_INCLINACION(punto1, punto2):
    """
    Calcula la inclinación de una línea definida por dos puntos.

    Parámetros:
        punto1: tuple - (x1, y1)
        punto2: tuple - (x2, y2)

    Retorna:
        float - la pendiente de la línea.

    Lanza:
        ValueError - si la línea es vertical (inclinación indefinida).
    """
    x1, y1 = punto1
    x2, y2 = punto2

    # Evita división por cero para líneas verticales.
    if x2 - x1 == 0:
        raise ValueError("La línea es vertical y la inclinación es indefinida.")

    return (y2 - y1) / (x2 - x1)

def FUN_CALCULAR_INCLINACION_EN_GRADOS(punto1, punto2):
    """
    Calcula el arco tangente (tan⁻¹) de x y lo retorna en grados.
    
    Parámetros:
    x (float): Número del cual se calculará el arco tangente.
    
    Retorna:
    float: Ángulo en grados correspondiente a tan⁻¹(x).
    """
    x = FUN_CALCULAR_INCLINACION(punto1,punto2)
    # Calcula el arco tangente en radianes y luego lo convierte a grados.
    return math.degrees(math.atan(x))

def FUN_OBTENER_MAXIMO_Y_COORDENADA(coordenadas):
    """
    Devuelve la tupla (x, y) con el valor más alto en y 
    de una lista de tuplas (x, y).
    """
    return max(coordenadas, key=lambda c: c[1])

def FUN_OBTENER_POSICION_MENTON(imagen_cv2):
    image = cv2.cvtColor(imagen_cv2, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    face_landmarks_list = face_recognition.face_landmarks(image)

    # Check if any faces are found
    if face_landmarks_list:
        # Extract chin landmarks (list of (x, y) coordinates)
        chin_points = face_landmarks_list[0]["chin"]
        menton_identificado = FUN_OBTENER_MAXIMO_Y_COORDENADA(chin_points)
        return menton_identificado
    else:
        return "No faces found in the image."

def FUN_CALCULAR_DISTANCIA_ENTRE_PUNTOS(punto_1, punto_2):
    x1, y1 = punto_1
    x2, y2 = punto_2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def FUN_CALCULAR_ANGULOS(A, B, C):
    """
    Calcula:
      - el ángulo entre el segmento AC y AB (ángulo en A)
      - el ángulo entre el segmento AC y BC (ángulo en C)
    
    Devuelve una tupla: (ángulo_AC_AB, ángulo_AC_BC)
    """
    # Calculamos las inclinaciones en A para los segmentos AC y AB
    inclinacion_AC = FUN_CALCULAR_INCLINACION_EN_GRADOS(A, C)
    inclinacion_AB = FUN_CALCULAR_INCLINACION_EN_GRADOS(A, B)
    
    # Ángulo en A es la diferencia absoluta entre las inclinaciones
    angulo_AC_AB = abs(inclinacion_AC - inclinacion_AB)
    # Si el ángulo es mayor de 90, tomamos el ángulo suplementario (ángulo agudo)
    if angulo_AC_AB > 90:
        angulo_AC_AB = 180 - angulo_AC_AB
    
    # Para el ángulo en C usamos AC y BC.
    # Se calculan las inclinaciones en C para los segmentos CA y CB.
    # (Recordar que la inclinación de CA es la misma que la de AC, pero con signo invertido)
    inclinacion_CA = FUN_CALCULAR_INCLINACION_EN_GRADOS(C, A)
    inclinacion_CB = FUN_CALCULAR_INCLINACION_EN_GRADOS(C, B)
    
    angulo_AC_BC = abs(inclinacion_CA - inclinacion_CB)
    if angulo_AC_BC > 90:
        angulo_AC_BC = 180 - angulo_AC_BC
    
    return angulo_AC_AB, angulo_AC_BC

def FUN_VALIDAR_INTEGRIDAD(pImagen: cv2) -> bool:
    """
    Valida la integridad de una imagen intentando redimensionarla

    :param pImagen: objeto de imagen OpenCV que se va a validar
    :return: True si la imagen se redimensiona correctamente, False si ocurre una excepción
    """
    try:
        cv2.resize(pImagen, (150, 150))
        return True
    except:
        return False

def exportar_a_excel(datos, directory):
    xlsx_files = [f for f in os.listdir(directory)
                  if os.path.isfile(os.path.join(directory, f)) and f.endswith('.xlsx')]
    id = len(xlsx_files) + 1
    df = pd.DataFrame(datos)
    df.to_excel(os.path.join(directory, f'validacion_postura_menton{id}.xlsx'), index=False)

# # Cargar la imagen de prueba
# ruta_imagen='/home/antho/Descargas/validated_color/1050452158.jpg'
# print(FUN_VALIDAR_POSICION_ROSTRO(cv2.imread(ruta_imagen)))


# # Rutas y procesamiento de imágenes
# ruta_carpeta_imagenes = "/home/antho/Descargas/validated_color"
# ruta_carpeta_excel = "/home/antho/Descargas/excel_validated_color"
# image_files = os.listdir(ruta_carpeta_imagenes)

# rutas_img = []
# # Posiciones_ojos = []
# larrVariacionPosicionMenton = []
# contador = 0  # Contador para las imágenes procesadas

# for image_file in image_files:
#     ruta_imagen_rel = os.path.join('../validated_color/', image_file)
#     hyperlink = f'=HYPERLINK("{ruta_imagen_rel}", "{image_file}")'
#     ruta_imagen = os.path.join(ruta_carpeta_imagenes, image_file)
#     imagen = cv2.imread(ruta_imagen)
#     if imagen is None:
#         continue
#     if FUN_VALIDAR_INTEGRIDAD(imagen):
#         rutas_img.append(hyperlink)
#         PosicionesOjos, InclinacionOjos = FUN_VALIDAR_POSICION_ROSTRO(imagen)
#         # Posiciones_ojos.append(PosicionesOjos)
#         larrVariacionPosicionMenton.append(InclinacionOjos)
# #     contador += 1  # Incrementa el contador después de procesar una imagen
# #     if contador % 3 == 0:
# #         segundos = 145
# #         print(f"Esperando {segundos} segundos después de procesar {contador} imágenes...")
# #         time.sleep(segundos)

# # Preparar y exportar los datos a Excel
# arreglo = [
#     ['Ruta', rutas_img],
#     # ['Posiciones de los ojos', Posiciones_ojos],
#     ['Inclinación', larrVariacionPosicionMenton]
# ]

# datos = {item[0]: item[1] for item in arreglo}
# exportar_a_excel(datos, ruta_carpeta_excel)

# ------------------- VALIDACION DEL MENTÓN ----------------------

# # Rutas y procesamiento de imágenes
# ruta_carpeta_imagenes = "/home/antho/Descargas/validated_color"
# ruta_carpeta_excel = "/home/antho/Descargas/excel_validated_color"
# image_files = os.listdir(ruta_carpeta_imagenes)

# rutas_img = []
# # Posiciones_ojos = []
# larrVariacionPosicionMenton = []
# contador = 0  # Contador para las imágenes procesadas

# for image_file in image_files:
#     ruta_imagen_rel = os.path.join('../validated_color/', image_file)
#     hyperlink = f'=HYPERLINK("{ruta_imagen_rel}", "{image_file}")'
#     ruta_imagen = os.path.join(ruta_carpeta_imagenes, image_file)
#     imagen = cv2.imread(ruta_imagen)
#     if imagen is None:
#         continue
#     if FUN_VALIDAR_INTEGRIDAD(imagen):
#         rutas_img.append(hyperlink)
#         lfloVariacionPosicionMenton = FUN_VALIDAR_POSICION_ROSTRO(imagen)
#         larrVariacionPosicionMenton.append(lfloVariacionPosicionMenton)
#     contador += 1  # Incrementa el contador después de procesar una imagen
#     if contador % 3 == 0:
#         segundos = 145
#         print(f"Esperando {segundos} segundos después de procesar {contador} imágenes...")
#         time.sleep(segundos)

# # Preparar y exportar los datos a Excel
# arreglo = [
#     ['Ruta', rutas_img],
#     # ['Posiciones de los ojos', Posiciones_ojos],
#     ['Variacion de hombros respecto al menton', larrVariacionPosicionMenton]
# ]

# datos = {item[0]: item[1] for item in arreglo}
# exportar_a_excel(datos, ruta_carpeta_excel)

# # Cargar la imagen de prueba
# ruta_imagen='/home/antho/Descargas/validated_color/1311307605.jpg'
# print(FUN_VALIDAR_POSTURA_ROSTRO(cv2.imread(ruta_imagen)))