import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

### FUNCIONES GENERALES REUTILIZABLES ###
def crear_carpeta(ruta):
    """Crea la carpeta si no existe."""
    os.makedirs(ruta, exist_ok=True)

def guardar_imagen(nombre, imagen, carpeta_salida):
    """Guarda una imagen en formato PNG en la carpeta de salida."""
    crear_carpeta(carpeta_salida)

    # Convertir imagen a uint8 si no lo es
    if imagen.dtype != np.uint8:
        imagen = np.clip(imagen, 0, 255).astype(np.uint8)

    # Si la imagen es en escala de grises (2D), la convierte en 3 canales para evitar errores en imsave
    if len(imagen.shape) == 2:
        imagen = np.stack([imagen] * 3, axis=-1)  # Convierte a RGB

    # Guardar la imagen correctamente
    ruta_imagen = os.path.join(carpeta_salida, f"{nombre}.png")
    plt.imsave(ruta_imagen, imagen)


### FUNCIONES PARA PRACTICA 1 ###
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

def imagen_escala_grises(tamano=255, ancho_barras=10):
    img = np.zeros((tamano, tamano), dtype=np.uint8)
    for i in range(0, tamano, ancho_barras):
        img[:, i:i + ancho_barras] = int((i / tamano) * 255)
    return img

def imagen_circulo_central(tamano=255, radio=80):
    img = np.zeros((tamano, tamano), dtype=np.uint8)
    centro = tamano // 2
    for i in range(tamano):
        for j in range(tamano):
            if (i - centro) ** 2 + (j - centro) ** 2 <= radio ** 2:
                img[i, j] = 255
    return img

def imagen_circulo_difuminado(tamano=255, radio=80):
    img = np.full((tamano, tamano), 255, dtype=np.uint8)
    centro = tamano // 2
    for i in range(tamano):
        for j in range(tamano):
            dist = np.sqrt((i - centro) ** 2 + (j - centro) ** 2)
            if dist <= radio:
                img[i, j] = int(255 * (1 - dist / radio))
    return img

def imagen_circulo_atenuado(tamano=255, radio=80):
    img = np.zeros((tamano, tamano), dtype=np.uint8)
    centro = tamano // 2
    for i in range(tamano):
        for j in range(tamano):
            dist = np.sqrt((i - centro) ** 2 + (j - centro) ** 2)
            if dist <= radio:
                img[i, j] = int(255 * (1 - dist / radio))
    return img

### FUNCIONES PARA PRACTICA 2 (PROCESAMIENTO DE IMÁGENES TIFF) ###
def normalizar_imagen(img, min_val=0, max_val=255):
    """Normaliza la imagen en un rango específico usando el min_val (0 por default) y max_val(255 por default)"""
    return cv2.normalize(img, None, min_val, max_val, cv2.NORM_MINMAX)

def ecualizar_imagen(img):
    """Aplica ecualización de histograma a una imagen en escala de grises"""
    return cv2.equalizeHist(img) #cv2.equalizeHist() solo funciona en imágenes de un canal (escala de grises)

def segmentar_histograma(nombre, canal, color, carpeta_salida):
    """Segmenta un histograma en 8 partes y lo grafica en un histograma de barras."""
    
    # Mapear nombres en español a colores válidos en Matplotlib
    color_map = {"azul": "blue", "verde": "green", "rojo": "red", "black": "black"}
    
    # Obtener color en inglés
    color = color_map.get(color, "black")

    # Calcular histograma
    histograma = cv2.calcHist([canal], [0], None, [256], [0, 256]).flatten()

    # Definir los 8 segmentos (cada uno cubre 32 valores de intensidad)
    segmentos = [sum(histograma[i * 32:(i + 1) * 32]) for i in range(8)]

    # Rango de valores para cada segmento
    etiquetas = ["0-31", "32-63", "64-95", "96-127", "128-159", "160-191", "192-223", "224-255"]

    # Graficar el histograma segmentado
    plt.figure(figsize=(8, 6))
    plt.bar(etiquetas, segmentos, color=color)
    plt.title(f'Segmentación del Histograma - {nombre}')
    plt.xlabel('Rangos de Intensidad')
    plt.ylabel('Frecuencia de Píxeles')
    plt.grid(axis='y')

    # Guardar la imagen del histograma segmentado
    ruta_segmento = os.path.join(carpeta_salida, f"{nombre}_segmentado.png")
    plt.savefig(ruta_segmento)
    plt.close()

    return segmentos  # Retorna los valores de cada segmento

def guardar_histograma(nombre, canal, color, carpeta_salida):
    """Calcula y guarda el histograma de un canal utilizando nombre, canal, color y carpeta_salida en ese orden"""
    # Mapear nombres en español a colores válidos en Matplotlib
    color_map = {"azul": "blue", "verde": "green", "rojo": "red"}
    color = color_map.get(color)
    histograma = cv2.calcHist([canal], [0], None, [256], [0, 256])
    plt.figure(figsize=(6, 4))
    plt.plot(histograma, color=color)
    plt.title(f'Histograma - {nombre}')
    plt.xlabel('Valor de Intensidad')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.savefig(os.path.join(carpeta_salida, f"{nombre}.png"))
    plt.close()
    
def aplicar_segmentacion_imagen(img):
    """Aplica la segmentación en 8 partes a la imagen y la devuelve con 8 niveles de intensidad."""

    # Definir los 8 niveles de intensidad correspondientes a cada segmento
    niveles_intensidad = [15, 45, 75, 105, 135, 165, 195, 225]  

    # Crear una imagen vacía con el mismo tamaño
    img_segmentada = np.zeros_like(img, dtype=np.uint8)

    # Asignar cada pixel al segmento correspondiente
    for i in range(8):
        rango_min = i * 32  # Límite inferior del segmento
        rango_max = (i + 1) * 32  # Límite superior del segmento
        img_segmentada[(img >= rango_min) & (img < rango_max)] = niveles_intensidad[i]

    return img_segmentada
