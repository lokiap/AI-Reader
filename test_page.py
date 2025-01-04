import cv2
import numpy as np

def test_connected_components(image_path):
    # Charger l'image et binariser
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Appliquer connectedComponentsWithStats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

    red_box_count = 0
    for i in range(num_labels):
        print(f"Composante {i} - Coordonnées (x, y): ({stats[i][0]}, {stats[i][1]})")
        #print(f"  - Largeur x Hauteur: {stats[i][2]} x {stats[i][3]}")
        #print(f"  - Aire: {stats[i][4]}")
        #print(f"  - Centroïde: ({centroids[i][0]:.2f}, {centroids[i][1]:.2f})")
        print("-" * 50)

         # Compter les composantes valides (aire > 50)
        if stats[i][4] > 50:
            red_box_count += 1

    print(f"Nombre total d'encadrés rouges : {red_box_count}")
# Afficher les résultats
    print(f"Nombre total de composantes détectées (y compris le fond) : {num_labels}\n")
    #print(f"Stats :\n{stats}\n")
    #print(f"Centroids :\n{centroids}\n")


if __name__ == "__main__":
    test_connected_components("Resultats/page.png")