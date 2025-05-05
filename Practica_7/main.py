import os
import cv2
import numpy as np
import sys
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functions import *

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
input_folder = "Practica_7/imagenes"
output_folder = "Practica_7/resultados"
crear_carpeta(output_folder)

# Lista de imágenes
image_names = ["garden_color.tiff", "Lena.tif", "test02.tif", "test09.tif", "test12.tif", "RGB_Image.png"]

# =============================================================================
# FUNCIÓN PARA PROCESAR UNA IMAGEN
# =============================================================================
def procesar_espacios_color(nombre_imagen, img_rgb, output_folder):
    """Procesa la imagen en RGB, HSV y CIELAB, separa canales y guarda resultados."""

    # ----- RGB -----
    B, G, R = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))  # OpenCV trabaja en BGR
    rgb_merge = cv2.merge((B, G, R))

    # Guardar canales RGB
    guardar_imagen(f"{nombre_imagen}_canal_R", R, output_folder, subcarpeta="RGB")
    guardar_imagen(f"{nombre_imagen}_canal_G", G, output_folder, subcarpeta="RGB")
    guardar_imagen(f"{nombre_imagen}_canal_B", B, output_folder, subcarpeta="RGB")
    guardar_imagen(f"{nombre_imagen}_imagen_RGB", rgb_merge, output_folder, subcarpeta="RGB")

    # ----- HSV -----
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(img_hsv)

    # Guardar canales HSV
    guardar_imagen(f"{nombre_imagen}_canal_H", H, output_folder, subcarpeta="HSV")
    guardar_imagen(f"{nombre_imagen}_canal_S", S, output_folder, subcarpeta="HSV")
    guardar_imagen(f"{nombre_imagen}_canal_V", V, output_folder, subcarpeta="HSV")
    guardar_imagen(f"{nombre_imagen}_imagen_HSV", img_hsv, output_folder, subcarpeta="HSV")

    # ----- CIELAB -----
    lab = convertir_rgb_a_cielab_manual(img_rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # Normalizar LAB para verlos
    L_norm = normalizar_imagen(L)
    a_norm = normalizar_imagen(a)
    b_norm = normalizar_imagen(b)

    # Guardar canales LAB
    guardar_imagen(f"{nombre_imagen}_componente_L", L_norm, output_folder, subcarpeta="CIELAB")
    guardar_imagen(f"{nombre_imagen}_componente_a", a_norm, output_folder, subcarpeta="CIELAB")
    guardar_imagen(f"{nombre_imagen}_componente_b", b_norm, output_folder, subcarpeta="CIELAB")

# =============================================================================
# PROCESAR TODAS LAS IMÁGENES
# =============================================================================
for name in image_names:
    if name == "RGB_Image.png":
        img_path = "Practica_7/imagenes/RGB_Image.png" 
    else:
        img_path = os.path.join(input_folder, name)

    print(f"Procesando {name}...")
    img_color = cv2.imread(img_path)
    if img_color is None:
        print(f"No se pudo leer {name}")
        continue
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # Convertir a RGB real

    procesar_espacios_color(os.path.splitext(name)[0], img_rgb, output_folder)

print("¡Todas las imágenes fueron procesadas en RGB, HSV y CIELAB!")