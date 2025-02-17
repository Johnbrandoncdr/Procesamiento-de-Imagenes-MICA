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

# Aplicar ecualización al histograma
canales_ecualizados = {color: ecualizar_imagen(canal) for color, canal in canales.items()}

# Guardar imágenes ecualizadas
for color, canal in canales_ecualizados.items():
    guardar_imagen(f"{nombre_imagen}_canal_{color}_ecualizado", canal, carpeta_salida)

# Unir los canales ecualizados y guardar la imagen resultante
imagen_ecualizada = cv2.merge((canales_ecualizados['azul'], canales_ecualizados['verde'], canales_ecualizados['rojo']))
guardar_imagen(f"{nombre_imagen}_imagen_ecualizada_tres_canales", imagen_ecualizada, carpeta_salida)

# Guardar histogramas de los canales originales
for color, canal in canales.items():
    guardar_histograma(f"{nombre_imagen}_histograma_{color}", canal, color, carpeta_salida)

# Guardar histogramas de los canales normalizados
for color, canal in canales_normalizados.items():
    guardar_histograma(f"{nombre_imagen}_histograma_{color}_normalizado", canal, color, carpeta_salida)

# Guardar histogramas de los canales ecualizados
for color, canal in canales_ecualizados.items():
    guardar_histograma(f"{nombre_imagen}_histograma_{color}_ecualizado", canal, color, carpeta_salida)

# Segmentar los histogramas en 8 partes
for color, canal in canales.items():
    segmentos = segmentar_histograma(f"{nombre_imagen}_segmento_{color}", canal, color, carpeta_salida)
    print(f"Segmentos del histograma de {color}: {segmentos}")

# Aplicar segmentación a la imagen en cada canal
canales_segmentados = {color: aplicar_segmentacion_imagen(canal) for color, canal in canales.items()}

# Guardar imágenes segmentadas
for color, canal in canales_segmentados.items():
    guardar_imagen(f"{nombre_imagen}_canal_{color}_segmentado", canal, carpeta_salida)

# Unir los canales segmentados y guardar la imagen resultante
imagen_segmentada = cv2.merge((canales_segmentados['azul'], canales_segmentados['verde'], canales_segmentados['rojo']))
guardar_imagen(f"{nombre_imagen}_imagen_segmentada", imagen_segmentada, carpeta_salida)

# Guardar histograma de la imagen final segmentada
guardar_histograma(f"{nombre_imagen}_histograma_imagen_segmentada", imagen_segmentada, 'black', carpeta_salida)

# Mostrar imágenes resultantes
cv2.imshow(f'{nombre_imagen} - Imagen Normalizada', imagen_normalizada)
cv2.imshow(f'{nombre_imagen} - Imagen Ecualizada', imagen_ecualizada)
cv2.imshow(f'{nombre_imagen} - Imagen Segmentada', imagen_segmentada)

for color, canal in canales.items():
    cv2.imshow(f'{nombre_imagen} - Canal {color.capitalize()}', canal)
    cv2.imshow(f'{nombre_imagen} - Canal {color.capitalize()} Segmentado', canales_segmentados[color])

cv2.waitKey(0)
cv2.destroyAllWindows()
