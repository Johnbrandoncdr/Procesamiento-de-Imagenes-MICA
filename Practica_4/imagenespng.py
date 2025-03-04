import cv2

# Cargar imágenes BMP
img_f_path = "Practica_4/imagenes//nuts_f.bmp"
img_g_path = "Practica_4/imagenes/nuts_g.bmp"

img_f = cv2.imread(img_f_path)
img_g = cv2.imread(img_g_path)

# Guardar imágenes en formato PNG
img_f_png_path = "Practica_4/resultados/nuts_f.png"
img_g_png_path = "Practica_4/resultados/nuts_g.png"

cv2.imwrite(img_f_png_path, img_f)
cv2.imwrite(img_g_png_path, img_g)

# Devolver las rutas de las imágenes convertidas
img_f_png_path, img_g_png_path
