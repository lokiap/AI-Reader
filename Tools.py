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


def Components_detection(image, binary_image, nom, y_tolerance=15):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image
    )

    # Charger l'image en couleur pour affichage
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    components = []

    # Taille de l'image pour éviter les composantes trop grandes (dans plan)
    img_h, img_w = image.shape

    # Collecter les composantes valides
    for i in range(1, num_labels):  # Ignorer le fond (0)
        x, y, w, h, area = stats[i]

        # Éviter les zones trop petites ou trop grandes
        if 50 < area < (0.5 * img_h * img_w) and w > 0 and h > 0:
            # Copie pour éviter les conflits
            roi = image[y : y + h, x : x + w].copy()
            components.append((roi, (x, y, w, h)))

    # Vérification avant de trier
    if not components:
        print("Aucune composante valide trouvée !")
        return color_image, []

    # Trier les composantes avec tolérance pour y (une fourchette y)
    sorted_components = []
    while components:
        base_component = components.pop(0)  # Récupérer et supprimer le premier élément
        line = [base_component]

        remaining_components = []  # Nouvelle liste pour les composants restants

        for component in components:
            # Vérifier que les dimensions sont cohérentes avant comparaison
            if len(component) != 2 or len(base_component) != 2:
                print(
                    f"Composante incorrecte ignorée : {(component[1][0],components[1][1])}"
                )
                continue

            # Comparer les positions en y pour regrouper sur une ligne
            if abs(component[1][1] - base_component[1][1]) <= y_tolerance:
                line.append(component)
            else:
                remaining_components.append(component)  # Garde les composants non liés

        # Mettre à jour les composants
        components = remaining_components

        # Trier les composants de la ligne horizontalement (par x)
        line.sort(key=lambda c: c[1][0])
        sorted_components.extend(line)

    # Dessiner les rectangles et afficher les numéros
    for idx, (roi, (x, y, w, h)) in enumerate(sorted_components):
        cv2.rectangle(
            color_image, (x, y), (x + w, y + h), (0, 0, 255), 2
        )  # Rectangle rouge
        if idx % 5 == 0:
            # Placer le texte à une hauteur constante (au-dessus de la composante)
            cv2.putText(
                color_image,
                f"{idx}",  # Numéro de composante
                (x, max(0, y - 10)),  # Fixer la hauteur à 10 pixels au-dessus
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # Taille du texte réduite
                (255, 0, 0),  # Couleur bleue
                2,
            )

    print(f"Nombre de composantes : {len(sorted_components)}")
    return color_image, sorted_components


# Retourne une composante précise
def get_Components(components, idx):
    return components[idx]


def display_components(component, nom):
    for i in range(len(component)):
        roi, (x, y, w, h) = component[i]
        # y = int(roi.shape[:2][0] + 10)
        # x = int(roi.shape[:2][1] / 2)
        # cv2.putText(
        #     roi,
        #     f"{nom[i]}",  # N
        #     (x, y),  # Fixer la hauteur à 10 pixels au-dessus
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,  # Taille du texte réduite
        #     (255, 0, 0),  # Couleur bleue
        #     2,
        # )
        cv2.imshow("CC", roi)
        # cv2.imshow(f"n°{i} : {nom[i]}", roi)
        cv2.waitKey(0)
        # cv2.destroyWindow(f"n°{i} : {nom[i]}")


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
    # Séparer l'expression dans une variable pour éviter l'erreur de parenthèses
    split_name = nom.split("/", 2)
    if len(split_name) > 2:
        cv2.imwrite(f"Resultats/{split_name[2]}", color_image)
    else:
        print(f"Erreur : la variable 'nom' ne contient pas assez de séparateurs '/'")
    cv2.destroyAllWindows()
