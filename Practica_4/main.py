import cv2 as cv
import numpy as np
import os

# Rutas de las imágenes
img_path = 'practica_4/imagenes/nuts_f.bmp'
template_path = 'practica_4/imagenes/nuts_g.bmp'

# Crear carpeta si no existe
resultados_dir = 'practica_4/resultados/'
if not os.path.exists(resultados_dir):
    os.makedirs(resultados_dir)

# Cargar imágenes en escala de grises
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)

# Convertir a float32 para la correlación
img = img.astype(np.float32)
template = template.astype(np.float32)

# Aplicar la correlación normalizada
result = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)

# Umbral de detección
threshold = 0.65  # Ajusta este valor según sea necesario
locations = np.where(result >= threshold)

# Convertir imagen a color para visualizar los círculos
img_color = cv.imread(img_path)  # Leer en color directamente
if img_color is None:
    print(f"Error: No se pudo cargar la imagen {img_path}")
    exit()

# Dibujar círculos en los puntos detectados
w, h = template.shape[::-1]  # Tamaño de la plantilla
for pt in zip(*locations[::-1]):  # Invertimos el orden de las coordenadas
    center = (pt[0] + w // 2, pt[1] + h // 2)
    cv.circle(img_color, center, radius=15, color=(0, 0, 255), thickness=2)

# Mostrar resultados
cv.imshow('Imagen Original con Detección', img_color)
cv.imshow('Resultado de la Correlación', result)
# Guardar las imágenes resultantes
cv.imwrite('practica_4/resultados/deteccion_nuts_f.png', img_color)
cv.imwrite('practica_4/resultados/correlacion_nuts_f.png', (result * 255).astype(np.uint8))  # Normaliza a 0-255

cv.waitKey(0)
cv.destroyAllWindows()
