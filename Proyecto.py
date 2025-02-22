import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Rutas de las imágenes en el aplicativo.
ruta_original = r"C:\Users\eduar\Documents\Proyecto_Detector\logotipo_original.png"
ruta_falsa = r"C:\Users\eduar\Documents\Proyecto_Detector\logotipo_falso.png"

# Cargar imágenes en escala de grises.
original = cv2.imread(ruta_original, cv2.IMREAD_GRAYSCALE)
falso = cv2.imread(ruta_falsa, cv2.IMREAD_GRAYSCALE)

# Para ejecutar y verificar si las imagenes se cargan correctamente.
if original is None:
    print("Error: No se pudo cargar logotipo_original.png")
    exit()
if falso is None:
    print("Error: No se pudo cargar logotipo_falso.png")
    exit()

# Redimensionar imágenes a 300x300, por si no encuentra el tamaño correcto
original = cv2.resize(original, (300, 300))
falso = cv2.resize(falso, (300, 300))

# Comparar las imágenes con SSIM
puntaje, diferencia = ssim(original, falso, full=True)
diferencia = (diferencia * 255).astype(np.uint8)

# Mapear las diferencias calcando las imagenes la falsa sobre la original.
_, umbral = cv2.threshold(diferencia, 50, 255, cv2.THRESH_BINARY)
contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar contornos en la imagen falsa para hallar las diferencias
falso_color = cv2.imread(ruta_falsa)
falso_color = cv2.resize(falso_color, (300, 300))
cv2.drawContours(falso_color, contornos, -1, (0, 0, 255), 2)

# Mostrar imágenes en distintas pestañas (Luego se modificara de otra manera la presentación)
cv2.imshow("Logotipo Original", original)
cv2.imshow("Logotipo Falso - Diferencias", falso_color)
cv2.imshow("Mapa de Diferencias", diferencia)

# Mensaje de verificación en la consola
if puntaje < 0.85:
    print(f"⚠️ Posible falsificación detectada. Similitud: {puntaje:.2f}⚠️")
else:
    print(f"✅El logotipo parece auténtico. Similitud: {puntaje:.2f}✅")

cv2.waitKey(0)
cv2.destroyAllWindows()
