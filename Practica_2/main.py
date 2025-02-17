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
imagen_color = cv2.imread(ruta_imagen)

if imagen_color is None:
    print("No se pudo cargar la imagen. Verifica la ruta.")
    exit()

# **Tomar solo el canal verde para convertirlo a escala de grises**
imagen_gris = imagen_color[:, :, 1]

# **Procesamiento Normalizado**
canales = {'azul': imagen_color[:, :, 0], 'verde': imagen_color[:, :, 1], 'rojo': imagen_color[:, :, 2]}
canales_normalizados = {color: normalizar_imagen(canal) for color, canal in canales.items()}
for color, canal in canales_normalizados.items():
    guardar_imagen(f"{nombre_imagen}_canal_{color}_normalizado", canal, carpeta_salida, "normalizada")
imagen_normalizada = cv2.merge(tuple(canales_normalizados.values()))
guardar_imagen(f"{nombre_imagen}_imagen_normalizada", imagen_normalizada, carpeta_salida, "normalizada")

# **Procesamiento Ecualizado**
canales_ecualizados = {color: ecualizar_imagen(canal) for color, canal in canales.items()}
for color, canal in canales_ecualizados.items():
    guardar_imagen(f"{nombre_imagen}_canal_{color}_ecualizado", canal, carpeta_salida, "ecualizada")
imagen_ecualizada = cv2.merge(tuple(canales_ecualizados.values()))
guardar_imagen(f"{nombre_imagen}_imagen_ecualizada", imagen_ecualizada, carpeta_salida, "ecualizada")

# **Segmentación (Cuantización)**
imagen_segmentada_4 = segmentar_imagen(imagen_gris, 4, carpeta_salida, nombre_imagen)
imagen_segmentada_8 = segmentar_imagen(imagen_gris, 8, carpeta_salida, nombre_imagen)
imagen_segmentada_16 = segmentar_imagen(imagen_gris, 16, carpeta_salida, nombre_imagen)

# **Guardar Histogramas**
for color, canal in canales.items():
    guardar_histograma(f"{nombre_imagen}_histograma_{color}", canal, color, carpeta_salida, "normalizada")
for color, canal in canales_ecualizados.items():
    guardar_histograma(f"{nombre_imagen}_histograma_{color}_ecualizado", canal, color, carpeta_salida, "ecualizada")

# **Guardar histograma segmentado de la imagen original**
guardar_histograma(f"{nombre_imagen}_histograma_original", imagen_gris, "gray", carpeta_salida, "segmentada")

# **Mostrar imágenes**
cv2.imshow('Imagen Normalizada', imagen_normalizada)
cv2.imshow('Imagen Ecualizada', imagen_ecualizada)
cv2.imshow('Imagen Original', imagen_gris)
cv2.imshow('Imagen Segmentada - 4 Niveles', imagen_segmentada_4)
cv2.imshow('Imagen Segmentada - 8 Niveles', imagen_segmentada_8)
cv2.imshow('Imagen Segmentada - 16 Niveles', imagen_segmentada_16)

cv2.waitKey(0)
cv2.destroyAllWindows()
