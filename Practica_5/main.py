import cv2 as cv
import numpy as np
import os
import sys
from PIL import Image

# Agregar la raíz del proyecto al path para importar functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functions import *


# Rutas
image_folder = "Practica_5/imagenes"
output_folder = "Practica_5/resultados"
crear_carpeta(output_folder)

D_VALUES = [15, 30, 60, 120]

image_files = ["Lena.tif", "polen.tif", "spine.tif"]

def load_tiff_image(image_path):
    img = Image.open(image_path).convert('L')
    return np.array(img, dtype=np.float32)

for img_name in image_files:
    img_path = os.path.join(image_folder, img_name)
    img = load_tiff_image(img_path)

    f, fshift = fourier_transform(img)

    for D0 in D_VALUES:
        # Aplicar Filtros
        img_lpf = inverse_fourier_transform(fshift * low_pass_filter(img.shape, D0))
        img_hpf = inverse_fourier_transform(fshift * high_pass_filter(img.shape, D0))
        img_bpf = inverse_fourier_transform(fshift * band_pass_filter(img.shape, D0, 20))
        img_brf = inverse_fourier_transform(fshift * band_reject_filter(img.shape, D0, 20))

        # Butterworth y Gaussiano
        img_butter_low = inverse_fourier_transform(fshift * butterworth_filter(img.shape, D0, 2, "low"))
        img_butter_high = inverse_fourier_transform(fshift * butterworth_filter(img.shape, D0, 2, "high"))
        img_gauss_low = inverse_fourier_transform(fshift * gaussian_filter(img.shape, D0, "low"))
        img_gauss_high = inverse_fourier_transform(fshift * gaussian_filter(img.shape, D0, "high"))

        # Crear y guardar las máscaras de los filtros
        mask_lpf = low_pass_filter(img.shape, D0)
        mask_hpf = high_pass_filter(img.shape, D0)
        mask_bpf = band_pass_filter(img.shape, D0, 20)
        mask_brf = band_reject_filter(img.shape, D0, 20)
        mask_butter_low = butterworth_filter(img.shape, D0, 2, "low")
        mask_butter_high = butterworth_filter(img.shape, D0, 2, "high")
        mask_gauss_low = gaussian_filter(img.shape, D0, "low")
        mask_gauss_high = gaussian_filter(img.shape, D0, "high")

        # Guardar las máscaras de los filtros
        guardar_imagen(f"{img_name}_Mask_LPF_{D0}", mask_lpf * 255, output_folder, "filtros")
        guardar_imagen(f"{img_name}_Mask_HPF_{D0}", mask_hpf * 255, output_folder, "filtros")
        guardar_imagen(f"{img_name}_Mask_BPF_{D0}", mask_bpf * 255, output_folder, "filtros")
        guardar_imagen(f"{img_name}_Mask_BRF_{D0}", mask_brf * 255, output_folder, "filtros")
        guardar_imagen(f"{img_name}_Mask_Butter_Low_{D0}", mask_butter_low * 255, output_folder, "filtros")
        guardar_imagen(f"{img_name}_Mask_Butter_High_{D0}", mask_butter_high * 255, output_folder, "filtros")
        guardar_imagen(f"{img_name}_Mask_Gauss_Low_{D0}", mask_gauss_low * 255, output_folder, "filtros")
        guardar_imagen(f"{img_name}_Mask_Gauss_High_{D0}", mask_gauss_high * 255, output_folder, "filtros")

        # Guardar Resultados
        guardar_imagen(f"{img_name}_LPF_{D0}", img_lpf, output_folder)
        guardar_imagen(f"{img_name}_HPF_{D0}", img_hpf, output_folder)
        guardar_imagen(f"{img_name}_BPF_{D0}", img_bpf, output_folder)
        guardar_imagen(f"{img_name}_BRF_{D0}", img_brf, output_folder)
        guardar_imagen(f"{img_name}_Butter_Low_{D0}", img_butter_low, output_folder)
        guardar_imagen(f"{img_name}_Butter_High_{D0}", img_butter_high, output_folder)
        guardar_imagen(f"{img_name}_Gauss_Low_{D0}", img_gauss_low, output_folder)
        guardar_imagen(f"{img_name}_Gauss_High_{D0}", img_gauss_high, output_folder)

print("Práctica 5 completada. Imágenes guardadas.")