import cv2
import os
import sys

# Agregar la raíz del proyecto al path para importar functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions import *

# Ruta de la imagen y carpeta de salida
ruta_imagen = "Practica_2/imagenes/spine.tif"
carpeta_salida = "Practica_2/histogramas"

# Extraer el nombre de la imagen sin extensión
nombre_imagen = os.path.splitext(os.path.basename(ruta_imagen))[0]

# Crear carpeta de salida si no existe
crear_carpeta(carpeta_salida)

# Cargar la imagen en color
imagen = cv2.imread(ruta_imagen)

# Verificar si la imagen se cargó correctamente
if imagen is None:
    print("No se pudo cargar la imagen. Verifica la ruta.")
    exit()

# Separar los canales de color
canales = {'azul': imagen[:, :, 0], 'verde': imagen[:, :, 1], 'rojo': imagen[:, :, 2]}

# Guardar imágenes separadas por color
for color, canal in canales.items():
    guardar_imagen(f"{nombre_imagen}_canal_{color}", canal, carpeta_salida)

# Normalizar cada canal en el rango [0, 255] por defecto en la función normalizar_imagen
canales_normalizados = {color: normalizar_imagen(canal) for color, canal in canales.items()}

# Guardar imágenes normalizadas
for color, canal in canales_normalizados.items():
    guardar_imagen(f"{nombre_imagen}_canal_{color}_normalizado", canal, carpeta_salida)

# Unir los canales normalizados y guardar la imagen resultante
imagen_normalizada = cv2.merge((canales_normalizados['azul'], canales_normalizados['verde'], canales_normalizados['rojo']))
guardar_imagen(f"{nombre_imagen}_imagen_normalizada_tres_canales", imagen_normalizada, carpeta_salida)

# Guardar histogramas de los canales originales
for color, canal in canales.items():
    guardar_histograma(f"{nombre_imagen}_histograma_{color}", canal, color, carpeta_salida)

# Guardar histogramas de los canales normalizados
for color, canal in canales_normalizados.items():
    guardar_histograma(f"{nombre_imagen}_histograma_{color}_normalizado", canal, color, carpeta_salida)

# Guardar histograma de la imagen final
guardar_histograma(f"{nombre_imagen}_histograma_imagen_normalizada_tres_canales", imagen_normalizada, '', carpeta_salida)

# Mostrar la imagen resultante y los canales separados
cv2.imshow(f'{nombre_imagen} - Imagen Normalizada', imagen_normalizada)
for color, canal in canales.items():
    cv2.imshow(f'{nombre_imagen} - Canal {color.capitalize()}', canal)

cv2.waitKey(0)
cv2.destroyAllWindows()

#Normalizacion y ecualización