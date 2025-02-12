import os
import sys

# Agregar la raíz del proyecto al path para importar functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions import *

# Carpeta de salida
carpeta_salida = "Practica_1/imagenes"

def generar_imagenes():
    imagenes = {
        "1_Blanca": imagen_blanca(),
        "2_Negra": imagen_negra(),
        "3_Barras": imagen_con_barras(),
        "4_Tablero_Ajedrez": imagen_tablero_ajedrez(),
        "5_Escala_Grises": imagen_escala_grises(),
        "6_Circulo_Central": imagen_circulo_central(),
        "7_Circulo_Difuminado": imagen_circulo_difuminado(),
        "8_Circulo_Atenuado": imagen_circulo_atenuado()
    }

    for nombre, img in imagenes.items():
        guardar_imagen(img, nombre, carpeta_salida)

if __name__ == "__main__":
    generar_imagenes()
    print(f"Imágenes generadas en '{carpeta_salida}'")
