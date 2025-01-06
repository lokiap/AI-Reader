import cv2
import os
from Tools import Choix, Gaussian_Threshold, Components_detection, Display

# Chargement du catalogue d'images
def load_catalogue(catalogue_path):
    catalogue = {}
    for filename in os.listdir(catalogue_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Charger l'image en nuance de gris
            image = cv2.imread(os.path.join(catalogue_path, filename), cv2.IMREAD_GRAYSCALE)
            catalogue[filename.split('.')[0]] = image  # Associer le nom sans extension à l'image
    return catalogue

# Comparaison d'un ROI avec le catalogue
def classify_component(roi, catalogue):
    best_match = None
    best_score = float('-inf')

    for name, ref_image in catalogue.items():
        # Redimensionner l'image de référence si nécessaire
        ref_resized = cv2.resize(ref_image, (roi.shape[1], roi.shape[0]))
        
        # Calculer la corrélation (Template Matching)
        result = cv2.matchTemplate(roi, ref_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        # Garder la meilleure correspondance
        if max_val > best_score:
            best_score = max_val
            best_match = name

    return best_match, best_score


    # Sauvegarder les composants dans leurs sous-dossiers
def save_components_by_label(components, catalogue, base_path="Resultats/lettres"):
    for i, (roi, (x, y, w, h)) in enumerate(components):
        label, score = classify_component(roi, catalogue)
        if score > 0.5:  # Seuil pour les correspondances acceptables
            subfolder_path = os.path.join(base_path, label)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)  # Créer le sous-dossier si inexistant
            
            # Sauvegarder l'image
            filename = f"component_{i}.png"
            filepath = os.path.join(subfolder_path, filename)
            cv2.imwrite(filepath, roi)
            print(f"Composant {i} sauvegardé dans {subfolder_path}")

# Exemple d'utilisation
if __name__ == "__main__":
    # Étape 1 : Charger et prétraiter l'image
    image, nom = Choix()  # Sélectionne l'image (1 - Page, 2 - Plan)
    binary_image = Gaussian_Threshold(image)  # Convertir en binaire
    
    # Étape 2 : Extraction des composants connectés
    color_image, components = Components_detection(image, binary_image, nom)

    # Chargement du catalogue
    catalogue_path = "data/caracteres/catalogue"  # Chemin vers les références
    catalogue = load_catalogue(catalogue_path)

    # Sauvegarder les composants dans leurs sous-dossiers respectifs
    save_components_by_label(components, catalogue, base_path="Resultats/lettres")
    
    # Classification des composants
    print(f"Nombre de composants détectés : {len(components)}")
    label_counts = {}
    for i, (roi, (x, y, w, h)) in enumerate(components):
        label, score = classify_component(roi, catalogue)
        if score > 0.5:  # Afficher uniquement les scores supérieurs à 0.8
            print(f"Composant {i} classé comme : {label} (Score : {score*100:.2f}%)")
            # Compter les occurrences des étiquettes
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

    # Affichage des occurrences
    print("\nNombre d'occurrences par étiquette (scores > 0.5) :")
    for label, count in label_counts.items():
        print(f"{label} : {count}")

    # Affichage de l'image avec les cadres
    Display(color_image, nom)
