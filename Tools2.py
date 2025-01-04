import cv2
import numpy as np


# Charger l'image fournie pour une nouvelle analyse
image_path = "data/caracteres/page.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Binariser l'image
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Appliquer cv2.connectedComponentsWithStats
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

# Compter les composantes (sans filtrage) et filtrer les petites composantes pour redbox
total_components = num_labels - 1  # Exclure le fond
red_box_count = 0
min_area = 50  # Seuil pour redbox

for i in range(1, num_labels):  # Ignorer le fond (composante 0)
    if stats[i][4] > min_area:  # Aire de la composante
        red_box_count += 1

total_components, red_box_count

# Afficher les résultats

print(f"Total des composantes détectées (sans filtrage) : {total_components}")
print(f"Nombre de composantes encadrées en rouge (aire > {min_area}) : {red_box_count}")