import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import color

# =============================================================================
# FUNCIONES GENERALES
# =============================================================================

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

# =============================================================================
# FUNCIONES PARA PRÁCTICA 2
# =============================================================================
  
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

# =============================================================================
# FUNCIONES DE TRANSFORMADA DE FOURIER
# =============================================================================

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

# =============================================================================
# FILTROS EN EL DOMINIO DE LA FRECUENCIA
# =============================================================================

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

# =============================================================================
# PRÁCTICA 7: RGB y CIELAB
# =============================================================================

def separar_canales_rgb(imagen):
    return cv2.split(imagen)

def convertir_a_grises(imagen_rgb):
    return cv2.cvtColor(imagen_rgb, cv2.COLOR_BGR2GRAY)

def umbralizar_multinivel(img_gray, niveles):
    max_val = 255
    thresholds = np.linspace(0, max_val, niveles + 1, dtype=np.uint8)[1:-1]
    img_umbral = np.zeros_like(img_gray)
    for i, thresh in enumerate(thresholds):
        img_umbral += ((img_gray > thresh) * (max_val // niveles)).astype(np.uint8)
    return img_umbral

def convertir_a_cielab(imagen_rgb):
    imagen_rgb_norm = imagen_rgb / 255.0
    return color.rgb2lab(imagen_rgb_norm)

def calcular_lab_promedio(imagen_lab, mascara):
    l = imagen_lab[..., 0][mascara]
    a = imagen_lab[..., 1][mascara]
    b = imagen_lab[..., 2][mascara]
    return np.mean(l), np.mean(a), np.mean(b)

def crear_imagen_seis_colores(lab_colores, shape):
    altura, ancho = shape
    imagen = np.zeros((altura, ancho, 3), dtype=np.float32)
    ancho_seccion = ancho // len(lab_colores)
    for i, color_lab in enumerate(lab_colores):
        imagen[:, i * ancho_seccion:(i + 1) * ancho_seccion, :] = color_lab
    imagen_rgb = color.lab2rgb(imagen) * 255
    return imagen_rgb.astype(np.uint8)

def calcular_lab_manual(imagen_rgb):
    """Convierte una imagen RGB a CIELAB de forma manual siguiendo las fórmulas del documento."""
    # 1. Normalizar a 0–1
    rgb = imagen_rgb.astype(np.float32) / 255.0

    # 2. Matriz de conversión RGB -> XYZ (espacio sRGB, iluminante D65)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])

    rgb = np.clip(rgb, 0, 1)
    shape = rgb.shape
    rgb_flat = rgb.reshape(-1, 3).T  # Transponer para multiplicar
    xyz_flat = M @ rgb_flat
    xyz = xyz_flat.T.reshape(shape)

    # 3. Normalización XYZ con blanco de referencia D65
    Xn, Yn, Zn = 1.0, 0.98072, 1.18225
    X = xyz[:, :, 0] / Xn
    Y = xyz[:, :, 1] / Yn
    Z = xyz[:, :, 2] / Zn

    # 4. Función f(t)
    def f(t):
        delta = 6 / 29
        return np.where(t > delta**3, t ** (1/3), (t / (3 * delta**2)) + (4 / 29))

    fx = f(X)
    fy = f(Y)
    fz = f(Z)

    L = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return L, a, b

def convertir_rgb_a_cielab_manual(imagen_rgb):
    imagen_rgb = imagen_rgb.astype(np.float32) / 255.0
    mask = imagen_rgb > 0.04045
    imagen_rgb[mask] = ((imagen_rgb[mask] + 0.055) / 1.055) ** 2.4
    imagen_rgb[~mask] /= 12.92
    imagen_rgb *= 100

    R, G, B = cv2.split(imagen_rgb)
    X = 0.4124564 * R + 0.3575761 * G + 0.1804375 * B
    Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B
    Z = 0.0193339 * R + 0.1191920 * G + 0.9503041 * B

    X /= 95.047
    Y /= 100.0
    Z /= 108.883

    def f(t):
        delta = 6/29
        return np.where(t > delta**3, t**(1/3), (t / (3 * delta**2)) + (4/29))

    fX = f(X)
    fY = f(Y)
    fZ = f(Z)

    L = (116 * fY) - 16
    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)

    lab = np.stack([L, a, b], axis=-1)
    return lab