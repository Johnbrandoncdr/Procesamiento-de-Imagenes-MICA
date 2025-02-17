import cv2
import os
import sys

# Agregar la raíz del proyecto al path para importar functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions import *

# Ruta de la imagen y carpeta de salida
ruta_imagen = "Practica_2/imagenes/Lena.tif"
carpeta_salida = "Practica_2/histogramas"

# Extraer el nombre de la imagen sin extensión
nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]

# Crear carpeta de salida si no existe
crear_carpeta(carpeta_salida)

# Cargar la imagen en color
imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print("No se pudo cargar la imagen. Verifica la ruta.")
    exit()

# Separar los canales de color
canales = {'azul': imagen[:, :, 0], 'verde': imagen[:, :, 1], 'rojo': imagen[:, :, 2]}

# Procesamiento Normalizado
canales_normalizados = {color: normalizar_imagen(canal) for color, canal in canales.items()}
for color, canal in canales_normalizados.items():
    guardar_imagen(f"{nombre_imagen}_canal_{color}_normalizado", canal, carpeta_salida, "normalizada")
imagen_normalizada = cv2.merge(tuple(canales_normalizados.values()))
guardar_imagen(f"{nombre_imagen}_imagen_normalizada", imagen_normalizada, carpeta_salida, "normalizada")

# Procesamiento Ecualizado
canales_ecualizados = {color: ecualizar_imagen(canal) for color, canal in canales.items()}
for color, canal in canales_ecualizados.items():
    guardar_imagen(f"{nombre_imagen}_canal_{color}_ecualizado", canal, carpeta_salida, "ecualizada")
imagen_ecualizada = cv2.merge(tuple(canales_ecualizados.values()))
guardar_imagen(f"{nombre_imagen}_imagen_ecualizada", imagen_ecualizada, carpeta_salida, "ecualizada")

# Procesamiento Segmentado
for color, canal in canales.items():
    aplicar_segmentacion_imagen(canal, carpeta_salida, f"{nombre_imagen}_canal_{color}")

# Guardar Histogramas
for color, canal in canales.items():
    guardar_histograma(f"{nombre_imagen}_histograma_{color}", canal, color, carpeta_salida, "normalizada")
for color, canal in canales_ecualizados.items():
    guardar_histograma(f"{nombre_imagen}_histograma_{color}_ecualizado", canal, color, carpeta_salida, "ecualizada")
guardar_histograma(f"{nombre_imagen}_histograma_imagen_segmentada", imagen, "black", carpeta_salida, "segmentada")

# Mostrar imágenes Normalizada y Ecualizada
cv2.imshow('Imagen Normalizada', imagen_normalizada)
cv2.imshow('Imagen Ecualizada', imagen_ecualizada)

cv2.waitKey(0)
cv2.destroyAllWindows()
