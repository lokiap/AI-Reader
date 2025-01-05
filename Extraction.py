import cv2
from Tools import Choix, Gaussian_Threshold, Components_detection, Display

# Étape 1 : Charger et prétraiter l'image
image, nom = Choix()  # Sélectionne l'image (1 - Page, 2 - Plan)
binary_image = Gaussian_Threshold(image)  # Convertit en binaire avec le seuillage adaptatif

# Étape 2 : Extraction des composants connectés
color_image, components = Components_detection(image, binary_image, nom)

# Étape 3 : Affichage des résultats
print(f"Nombre de composants détectés : {len(components)}")
for i, (roi, (x, y, w, h)) in enumerate(components):
    print(f"Composant {i + 1}: Coordonnées (x={x}, y={y}, w={w}, h={h})")

# Affichage de l'image avec les cadres
Display(color_image, nom)
