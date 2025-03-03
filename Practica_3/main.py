import cv2 as cv
import numpy as np
import os

"""
  Kernel	                          Descripción
Kernel 1 y 2	Detectores básicos de bordes en distintas direcciones.
Kernel 3 y 4	Filtros Sobel, útiles para la detección de bordes en X e Y.
Kernel 5 y 7	Filtros Laplacianos, resaltan cambios bruscos en la imagen.
Kernel 6 y 8	Variaciones del Laplaciano del Gaussiano, mejoran la detección de bordes finos.
Kernel 9	    Filtro de suavizado promedio, reduce el ruido.
Kernel 10	    Filtro Gaussiano, mejora la calidad reduciendo ruido sin perder bordes.
"""

# Carpeta de salida
carpeta_salida = "Practica_3/convolucion"
os.makedirs(carpeta_salida, exist_ok=True)

# Lista de imágenes
imagenes = [
    "aerial.tif", "einstein.tif", "kidney.tif", "Lena.tif",
    "pet.tif", "pollen.tif", "spine.tif", "test02.tif",
    "test09.tif", "test12.tif"
]

# Definición de las 10 máscaras de convolución
mascaras = {
    "kernel_1": np.array([[-1, 0, 1], [0, 1, 0], [0, 1, 0]]),
    "kernel_2": np.array([[0, -1, 0], [1, 0, 1], [0, 1, 0]]),
    "kernel_3": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    "kernel_4": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "kernel_5": np.array([[0, 0, 1], [1, -4, 1], [0, 0, 1]]),
    "kernel_6": np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]),
    "kernel_7": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
    "kernel_8": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    "kernel_9": (1/9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), 
    "kernel_10": (1/16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
}

# Aplicar la correlacion de las imagenes para cada mascara
for imagen_nombre in imagenes:
    ruta_imagen = f"Practica_3/imagenes/{imagen_nombre}"
    
    # Cargar la imagen en escala de grises
    img = cv.imread(ruta_imagen, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"No se pudo cargar la imagen {imagen_nombre}")
        continue

    # Aplicar cada uno de los 10 filtros y guardar las imágenes resultantes
    for nombre_kernel, kernel in mascaras.items():
        imagen_filtrada = cv.filter2D(img, -1, kernel)
        nombre_salida = f"{os.path.splitext(imagen_nombre)[0]}_{nombre_kernel}.png"
        ruta_salida = os.path.join(carpeta_salida, nombre_salida)
        cv.imwrite(ruta_salida, imagen_filtrada)
    
    print(f"Procesamiento de {imagen_nombre} completado.")

print("Convolución finalizada. Todas las imágenes han sido convolucionadas con cada mascara.")
