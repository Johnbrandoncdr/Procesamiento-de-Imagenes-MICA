import os
import numpy as np
import matplotlib.pyplot as plt

# Carpeta donde se van a guardar las imágenes finales
carpeta_salida = "Practica_1/imagenes"
os.makedirs(carpeta_salida, exist_ok=True)

def imagen_blanca(tamano=255): 
    """ Genera una imagen completamente blanca de tamaño tamano x tamano """
    return np.full((tamano, tamano), 255, dtype=np.uint8)

def imagen_negra(tamano=255):
    """ Genera una imagen completamente negra de tamaño tamano x tamano """
    return np.zeros((tamano, tamano), dtype=np.uint8)

def imagen_con_barras(tamano=255, ancho_barras=10):
    """ Genera una imagen con barras verticales blancas y negras alternadas """
    img = np.zeros((tamano, tamano), dtype=np.uint8)
    for i in range(0, tamano, ancho_barras * 2): 
        img[:, i:i + ancho_barras] = 255  # Se alternan las barras blancas
    return img

def imagen_tablero_ajedrez(tamano=255, tamano_cuadro=20):
    """ Genera una imagen con un patrón de tablero de ajedrez """
    img = np.zeros((tamano, tamano), dtype=np.uint8)
    for i in range(0, tamano, tamano_cuadro * 2): 
        for j in range(0, tamano, tamano_cuadro * 2): 
            img[i:i + tamano_cuadro, j:j + tamano_cuadro] = 255  # Bloque blanco
            img[i + tamano_cuadro:i + 2 * tamano_cuadro, j + tamano_cuadro:j + 2 * tamano_cuadro] = 255  # Bloque desplazado
    return img

def imagen_escala_grises(tamano=255):
    """ Genera una imagen con una transición de negro a blanco en escala de grises """
    img = np.zeros((tamano, tamano), dtype=np.uint8)
    for i in range(tamano):
        img[:, i] = int((i / tamano) * 255)  # De negro (0) a blanco (255)
    return img

def imagen_circulo_central(tamano=255, radio=80):
    """ Genera una imagen con un círculo blanco en el centro """
    img = np.zeros((tamano, tamano), dtype=np.uint8)
    centro = tamano // 2
    for i in range(tamano):
        for j in range(tamano):
            if (i - centro) ** 2 + (j - centro) ** 2 <= radio ** 2:
                img[i, j] = 255  # Se dibuja el círculo blanco
    return img

def imagen_circulo_difuminado(tamano=255, radio=80):
    """ Genera una imagen con un círculo con un degradado hacia negro """
    img = np.full((tamano, tamano), 255, dtype=np.uint8)
    centro = tamano // 2
    for i in range(tamano):
        for j in range(tamano):
            dist = np.sqrt((i - centro) ** 2 + (j - centro) ** 2)
            if dist <= radio:
                img[i, j] = int(255 * (1- dist / radio))  # Atenuación dentro del círculo (blanco al centro, negro al borde) lo demas blanco
    
    return img

def imagen_circulo_atenuado(tamano=255, radio=80):
    """ Genera una imagen con un círculo que se desvanece de blanco a negro """
    img = np.zeros((tamano, tamano), dtype=np.uint8)
    centro = tamano // 2
    for i in range(tamano):
        for j in range(tamano):
            dist = np.sqrt((i - centro) ** 2 + (j - centro) ** 2)
            if dist <= radio:
                img[i, j] = int(255 * (1 - dist / radio))  # Atenuación dentro del círculo (blanco al centro, negro al borde) lo demas negro
    return img

def guardar_imagen(imagen, nombre):
    """ Guarda la imagen en formato PNG, asegurando que las imágenes blancas no se guarden como negras """
    if imagen.ndim == 2:  # Convertir imágenes en escala de grises a RGB para evitar errores de visualización
        imagen = np.stack([imagen] * 3, axis=-1)
    plt.imsave(f"{carpeta_salida}/{nombre}.png", imagen, cmap='gray', vmin=0, vmax=255)
    """ Guarda la imagen en formato PNG """
    plt.imsave(f"{carpeta_salida}/{nombre}.png", imagen, cmap='gray', vmin=0, vmax=255)

def generar_imagenes():
    """ Genera y guarda todas las imágenes """
    guardar_imagen(imagen_blanca(), "1_Blanca")
    guardar_imagen(imagen_negra(), "2_Negra")
    guardar_imagen(imagen_con_barras(), "3_Barras")
    guardar_imagen(imagen_tablero_ajedrez(), "4_Tablero_Ajedrez")
    guardar_imagen(imagen_escala_grises(), "5_Escala_Grises")
    guardar_imagen(imagen_circulo_central(), "6_Circulo_Central")
    guardar_imagen(imagen_circulo_difuminado(), "7_Circulo_Difuminado")
    guardar_imagen(imagen_circulo_atenuado(), "8_Circulo_Atenuado")

if __name__ == "__main__":
    generar_imagenes()
    print("Imágenes generadas y guardadas en la carpeta 'imagenes'")