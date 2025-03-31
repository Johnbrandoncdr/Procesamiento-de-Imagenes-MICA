import os
import cv2
import numpy as np
import sys
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functions import *

# =============================================================================
# CONFIGURACIÓN INICIAL
# =============================================================================
input_folder = "Practica_7/imagenes"
output_folder = "Practica_7/resultados"
crear_carpeta(output_folder)

image_names = ["garden_color.tiff", "Lena.tif", "test02.tif", "test09.tif", "test12.tif"]
colores_lab = [
    (0,   0,   0),     # Negro
    (100, 0,   0),     # Blanco
    (53,  80,  67),    # Rojo
    (87, -86, 83),     # Verde
    (32, 79, -108),    # Azul
    (97, -21, 94),     # Amarillo
]

# =============================================================================
# PARTE 1: CREACIÓN DE IMAGEN DE 6 COLORES Y CÁLCULO DE CIELAB
# =============================================================================
print("Procesando imagen con 6 colores...")

img_colores = crear_imagen_seis_colores(colores_lab, shape=(100, 600))
guardar_imagen("imagen_6_colores_rgb", img_colores, output_folder, subcarpeta="colores")

# Convertir a CIELAB (manual)
L, a, b = calcular_lab_manual(img_colores)

# Normalizar a 0–255 y guardar
L_norm = normalizar_imagen(L)
a_norm = normalizar_imagen(a)
b_norm = normalizar_imagen(b)

guardar_imagen("imagen_6_colores_L", L_norm, output_folder, subcarpeta="colores")
guardar_imagen("imagen_6_colores_a", a_norm, output_folder, subcarpeta="colores")
guardar_imagen("imagen_6_colores_b", b_norm, output_folder, subcarpeta="colores")

# =============================================================================
# PARTE 2: PROCESAMIENTO DE ESCENA NATURAL
# =============================================================================
for name in image_names:
    print(f"Procesando imagen: {name}")
    img_path = os.path.join(input_folder, name)
    img_color = cv2.imread(img_path)
    if img_color is None:
        print(f"No se pudo leer {name}")
        continue

    # Convertir a CIELAB
    lab = convertir_rgb_a_cielab_manual(img_color)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # Normalizar a 0–255
    L_norm = normalizar_imagen(L)
    a_norm = normalizar_imagen(a)
    b_norm = normalizar_imagen(b)

    # Guardar imágenes L, a, b
    guardar_imagen(f"{name}_L", L_norm, output_folder, subcarpeta="escena")
    guardar_imagen(f"{name}_a", a_norm, output_folder, subcarpeta="escena")
    guardar_imagen(f"{name}_b", b_norm, output_folder, subcarpeta="escena")

    # Umbralización multinivel en cada componente
    L_umb = umbralizar_multinivel(L_norm, 5)
    a_umb = umbralizar_multinivel(a_norm, 5)
    b_umb = umbralizar_multinivel(b_norm, 5)

    guardar_imagen(f"{name}_L_umbral", L_umb, output_folder, subcarpeta="escena")
    guardar_imagen(f"{name}_a_umbral", a_umb, output_folder, subcarpeta="escena")
    guardar_imagen(f"{name}_b_umbral", b_umb, output_folder, subcarpeta="escena")

print("Práctica 7 completada.")
