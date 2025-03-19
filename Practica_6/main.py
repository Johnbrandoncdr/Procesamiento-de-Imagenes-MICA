import cv2 as cv
import numpy as np

# Inicializar la captura de video desde la cámara (0 para la cámara integrada)
cap = cv.VideoCapture(0)

# Definir un kernel para convolución (puedes modificarlo según necesites)
kernel = np.array([[ -1,  0,  1], 
                   [ -2,  0,  2], 
                   [ -1,  0,  1]])

# Ciclo infinito para leer la cámara en tiempo real
while cap.isOpened():
    # Captura el frame
    ret, frame = cap.read()
    
    # Si no se capturó correctamente, salir del bucle
    if not ret:
        print("No se pudo capturar el video")
        break
    
    # Convertir el frame a escala de grises
    img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Aplicar filtro de convolución
    img_convolucionada = cv.filter2D(img_gray, -1, kernel)
    
    # Aplicar filtro Sobel en X y Y
    sobelx = cv.Sobel(img_gray, cv.CV_64F, 1, 0, ksize=3)  # Derivada en X
    sobely = cv.Sobel(img_gray, cv.CV_64F, 0, 1, ksize=3)  # Derivada en Y
    sobelxy = cv.sqrt(sobelx**2 + sobely**2)  # Magnitud combinada
    
    # Aplicar el filtro Canny para detección de bordes
    canny = cv.Canny(img_gray, 100, 200)
    
    # Mostrar los resultados
    cv.imshow('Imagen Original (Gris)', img_gray)
    cv.imshow('Filtro Convolución', img_convolucionada)
    cv.imshow('Sobel XY', sobelxy.astype(np.uint8))  # Convertir a uint8 para mostrar
    cv.imshow('Canny', canny)
    
    # Salir con la tecla ESC (código ASCII 27)
    if cv.waitKey(1) == 27:
        break

# Liberar los recursos y cerrar ventanas
cap.release()
cv.destroyAllWindows()
