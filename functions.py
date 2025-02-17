import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

### FUNCIONES GENERALES ###
def crear_carpeta(ruta):
    """Crea la carpeta si no existe."""
    os.makedirs(ruta, exist_ok=True)

def guardar_imagen(nombre, imagen, carpeta_salida, subcarpeta=""):
    """Guarda una imagen en formato PNG en la subcarpeta correspondiente."""
    ruta_completa = os.path.join(carpeta_salida, subcarpeta)
    crear_carpeta(ruta_completa)

    # Convertir imagen a uint8 si no lo es
    if imagen.dtype != np.uint8:
        imagen = np.clip(imagen, 0, 255).astype(np.uint8)

    # Si la imagen es en escala de grises (2D), convertirla a RGB
    if len(imagen.shape) == 2:
        imagen = np.stack([imagen] * 3, axis=-1)

    ruta_imagen = os.path.join(ruta_completa, f"{nombre}.png")
    cv2.imwrite(ruta_imagen, imagen)

def guardar_histograma(nombre, canal, color, carpeta_salida, subcarpeta=""):
    """Calcula y guarda el histograma en la subcarpeta correspondiente."""
    ruta_completa = os.path.join(carpeta_salida, subcarpeta)
    crear_carpeta(ruta_completa)

    # Mapear nombres en español a colores válidos en Matplotlib
    color_map = {"azul": "blue", "verde": "green", "rojo": "red", "black": "black"}
    color = color_map.get(color, "black")

    histograma = cv2.calcHist([canal], [0], None, [256], [0, 256])

    plt.figure(figsize=(6, 4))
    plt.plot(histograma, color=color)
    plt.title(f'Histograma - {nombre}')
    plt.xlabel('Valor de Intensidad')
    plt.ylabel('Frecuencia')
    plt.grid(True)

    ruta_histograma = os.path.join(ruta_completa, f"{nombre}.png")
    plt.savefig(ruta_histograma)
    plt.close()

### FUNCIONES PARA PRÁCTICA 1 ###
def imagen_blanca(tamano=255): 
    return np.full((tamano, tamano), 255, dtype=np.uint8)

def imagen_negra(tamano=255):
    return np.zeros((tamano, tamano), dtype=np.uint8)

def imagen_con_barras(tamano=255, ancho_barras=10):
    img = np.zeros((tamano, tamano), dtype=np.uint8)
    for i in range(0, tamano, ancho_barras * 2): 
        img[:, i:i + ancho_barras] = 255
    return img

def imagen_tablero_ajedrez(tamano=255, tamano_cuadro=20):
    img = np.zeros((tamano, tamano), dtype=np.uint8)
    for i in range(0, tamano, tamano_cuadro * 2): 
        for j in range(0, tamano, tamano_cuadro * 2): 
            img[i:i + tamano_cuadro, j:j + tamano_cuadro] = 255
            img[i + tamano_cuadro:i + 2 * tamano_cuadro, j + tamano_cuadro:j + 2 * tamano_cuadro] = 255
    return img

### FUNCIONES PARA PRÁCTICA 2 ###
def normalizar_imagen(img, min_val=0, max_val=255):
    """Normaliza la imagen en un rango específico."""
    return cv2.normalize(img, None, min_val, max_val, cv2.NORM_MINMAX)

def ecualizar_imagen(img):
    """Aplica ecualización de histograma a una imagen en escala de grises."""
    return cv2.equalizeHist(img)

def segmentar_histograma(nombre, imagen, carpeta_salida, subcarpeta="segmentada"):
    """Segmenta el histograma de una imagen completa o de un canal en 8 partes y lo guarda."""

    ruta_completa = os.path.join(carpeta_salida, subcarpeta)
    crear_carpeta(ruta_completa)

    histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()

    # Dividir el histograma en 8 segmentos de 32 valores cada uno
    segmentos = [sum(histograma[i * 32:(i + 1) * 32]) for i in range(8)]

    etiquetas = ["0-31", "32-63", "64-95", "96-127", "128-159", "160-191", "192-223", "224-255"]

    # Graficar el histograma segmentado
    plt.figure(figsize=(8, 6))
    plt.bar(etiquetas, segmentos, color="gray")  # Color gris para la imagen original
    plt.title(f'Segmentación del Histograma - {nombre}')
    plt.xlabel('Rangos de Intensidad')
    plt.ylabel('Frecuencia de Píxeles')
    plt.grid(axis='y')

    ruta_segmento = os.path.join(ruta_completa, f"{nombre}_segmentado.png")
    plt.savefig(ruta_segmento)
    plt.close()

    return segmentos

def aplicar_segmentacion_imagen(img, carpeta_salida, nombre_imagen, subcarpeta="segmentada"):
    """Genera 8 imágenes segmentadas, cada una mostrando un rango específico de intensidad."""
    
    ruta_completa = os.path.join(carpeta_salida, subcarpeta)
    crear_carpeta(ruta_completa)

    niveles_intensidad = [15, 45, 75, 105, 135, 165, 195, 225]  # Intensidad representativa de cada segmento
    imagenes_segmentadas = []  # Lista para almacenar las imágenes generadas

    for i in range(8):
        rango_min = i * 32
        rango_max = (i + 1) * 32

        # Crear una máscara binaria para el rango actual
        img_segmentada = np.zeros_like(img, dtype=np.uint8)
        img_segmentada[(img >= rango_min) & (img < rango_max)] = niveles_intensidad[i]

        # Guardar la imagen segmentada
        guardar_imagen(f"{nombre_imagen}_segmento_{rango_min}-{rango_max}", img_segmentada, carpeta_salida, subcarpeta)

        # Agregar a la lista
        imagenes_segmentadas.append(img_segmentada)

    return imagenes_segmentadas
