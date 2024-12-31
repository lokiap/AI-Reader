import cv2


# Choix de l'image
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

# Image binaire
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

# Détection des composantes connexes
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

# Charger l'image en couleur pour affichage des encadrements
color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Parcours des objets détectés pour les encadrer
for i in range(1, num_labels):  # On ignore le fond même s'il est blanc (le 0)
    x, y, w, h, area = stats[i]
    if area > 50:  # Filtrer SEULEMENT les petits objets
        cv2.rectangle(
            color_image, (x, y), (x + w, y + h), (0, 0, 255), 2
        )  # Ca encadre en rouge


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
max_width = 1000
max_height = 800


# Calculer le facteur d'échelle pour ajuster la taille

resized_image = resize_with_aspect_ratio(
    color_image, width=max_width, height=max_height
)

# Affichage de l'image encadrée
cv2.imshow("Resultat", resized_image)
cv2.waitKey(0)
cv2.imwrite(f'Resultats/{nom.split('/',2)[2]}', color_image)
cv2.destroyAllWindows()
