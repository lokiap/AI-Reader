import cv2
import numpy as np


# Choix de l'image
def Choix():
    nom = input("Quel fichier à analyser ? 1 - Page || 2 - Plan :\n")
    while True:
        if nom == "1":
            nom = "data/caracteres/page.png"
            break
        elif nom == "2":
            nom = "data/plans/plan.png"
            break
        nom = input("Veuillez répondre 1 ou 2 :\n")

    # Lecture de l'image (en nuance de gris)
    image = cv2.imread(nom, cv2.IMREAD_GRAYSCALE)
    return image, nom


# Image binaire
def Gaussian_Threshold(image):
    binary_image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )  # 11 => espace autour du pixel, 2 => Constante pour jouer avec la moyenne
    return binary_image


def Mean_Threshold(image):
    binary_image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return binary_image


def Threshold(image):
    binary_image = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY_INV)
    return binary_image


# Détection des composantes connexes
def Components_detection(image, binary_image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image
    )

    # Charger l'image en couleur pour affichage des encadrements
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    components = []
    # Parcours des objets détectés pour les encadrer
    for i in range(1, num_labels):  # On ignore le fond même s'il est blanc (le 0)
        x, y, w, h, area = stats[i]
        roi = image[y : y + h, x : x + w]
        if area > 50:  # Filtrer SEULEMENT les petits objets
            components.append((roi, (x, y, w, h)))
            cv2.rectangle(
                color_image, (x, y), (x + w, y + h), (0, 0, 255), 2
            )  # Ca encadre en rouge
    return color_image, components


# Fonction pour ajuster l'image à la fenêtre tout en conservant les proportions
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


# Redimensionner l'image pour une meilleure visualisation à l'ouverture
def Display(color_image, nom):
    max_width = 1000
    max_height = 800
    resized_image = resize_with_aspect_ratio(
        color_image, width=max_width, height=max_height
    )

    # Affichage de l'image encadrée
    cv2.imshow("Resultat", resized_image)
    cv2.waitKey(0)
    cv2.imwrite(f'Resultats/{nom.split('/',2)[2]}', color_image)
    cv2.destroyAllWindows()
