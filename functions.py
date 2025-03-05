import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

### FUNCIONES GENERALES ###
def crear_carpeta(ruta):
    """Crea la carpeta si no existe."""
    os.makedirs(ruta, exist_ok=True)

def guardar_imagen(nombre, imagen, carpeta_salida, subcarpeta="segmentada"):
    """Guarda una imagen en formato PNG en la carpeta de salida."""
    ruta_completa = os.path.join(carpeta_salida, subcarpeta)
    crear_carpeta(ruta_completa)

    # Convertir imagen a uint8 si no lo es
    if imagen.dtype != np.uint8:
        imagen = np.clip(imagen, 0, 255).astype(np.uint8)

    # Guardar la imagen
    ruta_imagen = os.path.join(ruta_completa, f"{nombre}.png")
    cv2.imwrite(ruta_imagen, imagen)

def guardar_histograma(nombre, canal, color, carpeta_salida, subcarpeta=""):
    """Calcula y guarda el histograma en la subcarpeta correspondiente."""
    ruta_completa = os.path.join(carpeta_salida, subcarpeta)
    crear_carpeta(ruta_completa)

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

### FUNCIONES PARA PRÁCTICA 2 ###
def normalizar_imagen(img, min_val=0, max_val=255):
    """Normaliza la imagen en un rango específico."""
    return cv2.normalize(img, None, min_val, max_val, cv2.NORM_MINMAX)

def ecualizar_imagen(img):
    """Aplica ecualización de histograma a una imagen en escala de grises."""
    return cv2.equalizeHist(img)

def segmentar_imagen(img, niveles, carpeta_salida, nombre_imagen):
    """Reduce la imagen a una cantidad específica de niveles de gris y la guarda (cuantización)."""
    ruta_completa = os.path.join(carpeta_salida, "segmentada")
    crear_carpeta(ruta_completa)

    # Determinar el factor de cuantización
    factor = 256 // niveles

    # Aplicar la segmentación
    img_segmentada = (img // factor) * factor

    # Guardar la imagen segmentada
    guardar_imagen(f"{nombre_imagen}_segmentada_{niveles}_niveles", img_segmentada, carpeta_salida, "segmentada")

    return img_segmentada

### FUNCIONES DE TRANSFORMADA DE FOURIER ###
def fourier_transform(image):
    """Aplica la Transformada de Fourier y centra el espectro."""
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    return f, fshift

def inverse_fourier_transform(fshift):
    """Aplica la Transformada Inversa de Fourier."""
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)

### FILTROS EN EL DOMINIO DE LA FRECUENCIA ###
def low_pass_filter(shape, D0):
    """Genera un filtro pasa bajas ideal."""
    M, N = shape
    H = np.zeros((M, N), dtype=np.float32)
    center = (M // 2, N // 2)

    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - center[0])**2 + (v - center[1])**2)
            if D <= D0:
                H[u, v] = 1
    return H

def high_pass_filter(shape, D0):
    """Genera un filtro pasa altas ideal."""
    return 1 - low_pass_filter(shape, D0)

def band_pass_filter(shape, D0, W):
    """Genera un filtro pasa bandas ideal."""
    return low_pass_filter(shape, D0 + W // 2) - low_pass_filter(shape, D0 - W // 2)

def band_reject_filter(shape, D0, W):
    """Genera un filtro rechazo de banda ideal."""
    return 1 - band_pass_filter(shape, D0, W)

def butterworth_filter(shape, D0, n, type="low"):
    """Genera un filtro Butterworth (pasa bajas o pasa altas)."""
    M, N = shape
    H = np.zeros((M, N), dtype=np.float32)
    center = (M // 2, N // 2)

    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - center[0])**2 + (v - center[1])**2)
            if type == "low":
                H[u, v] = 1 / (1 + (D / D0) ** (2 * n))
            elif type == "high":
                H[u, v] = 1 - 1 / (1 + (D / D0) ** (2 * n))
    
    return H

def gaussian_filter(shape, D0, type="low"):
    """Genera un filtro Gaussiano (pasa bajas o pasa altas)."""
    M, N = shape
    H = np.zeros((M, N), dtype=np.float32)
    center = (M // 2, N // 2)

    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - center[0])**2 + (v - center[1])**2)
            H[u, v] = np.exp(-(D**2) / (2 * (D0**2)))
    
    if type == "high":
        return 1 - H
    return H