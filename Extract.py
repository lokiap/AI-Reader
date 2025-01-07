import cv2
import numpy as np
import os
import Tools


def extract_hu_moments(image):
    """Extrait les moments de Hu de l'image binaire."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    return np.log1p(np.abs(hu_moments))  # Compression de l'échelle avec log(1 + abs(x))


def extract_histogram(image, bins=32):
    """Extrait un histogramme normalisé des couleurs ou des niveaux de gris."""
    if len(image.shape) == 3:  # Image couleur
        hist = []
        for i in range(3):  # Canaux B, G, R
            hist_channel = cv2.calcHist([image], [i], None, [bins], [0, 256])
            hist.append(hist_channel.flatten())
        hist = np.concatenate(hist)
    else:  # Image en niveaux de gris
        hist = cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()
    return hist / np.sum(hist)  # Normalisation de l'histogramme


def normalize_features(features):
    """Normalise les caractéristiques pour qu'elles aient une moyenne de 0 et un écart-type de 1."""
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / (
        std + 1e-8
    )  # Évite la division par 0 avec un petit epsilon


def extract_features_from_rois(rois):
    """Extrait et combine les caractéristiques des ROIs (moments de Hu + histogrammes)."""
    features = []
    for roi, coords in rois:
        x, y, w, h = coords
        hu_moments = extract_hu_moments(roi)  # 7 moments
        histogram = extract_histogram(roi, bins=32)  # 32 valeurs
        combined_features = np.concatenate([hu_moments, histogram])  # Taille : 39
        features.append(combined_features)
    features_array = np.array(features)

    # Vérification des dimensions
    print(f"Features from ROIs shape: {features_array.shape}")
    return features_array


def process_reference_images(reference_directory):
    """
    Charge les images de référence depuis un dossier,
    extrait leurs caractéristiques, et leur associe un label.
    """
    reference_features = []
    labels = []
    for filename in os.listdir(reference_directory):
        file_path = os.path.join(reference_directory, filename)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            # Extraire les caractéristiques de l'image de référence
            hu_moments = extract_hu_moments(image)  # 7 moments

            print(f"Hu moments shape: {hu_moments.shape}")
            histogram = extract_histogram(image, bins=32)  # 32 valeurs

            print(f"Histogram shape: {histogram.shape}")
            combined_features = np.concatenate([hu_moments, histogram])  # Taille : 39
            reference_features.append(combined_features)
            labels.append(filename)  # Le nom du fichier peut servir de label

    reference_features_array = np.array(reference_features)

    # Vérification des dimensions
    print(f"Reference features shape: {reference_features_array.shape}")
    return reference_features_array, labels


def process_data(rois, reference_directory):
    """
    1. Extrait les caractéristiques des ROIs.
    2. Extrait les caractéristiques des images de référence.
    3. Normalise toutes les caractéristiques ensemble.
    """
    reference_directory = f"{reference_directory}/catalogue"
    # Extraire les caractéristiques des ROIs
    roi_features = extract_features_from_rois(rois)

    # Extraire les caractéristiques des images de référence
    reference_features, reference_labels = process_reference_images(reference_directory)

    # Combiner les caractéristiques des ROIs et des références pour normalisation
    all_features = np.vstack([roi_features, reference_features])
    normalized_features = normalize_features(all_features)

    # Séparer les caractéristiques normalisées
    normalized_roi_features = normalized_features[: len(roi_features)]
    normalized_reference_features = normalized_features[len(roi_features) :]

    return normalized_roi_features, normalized_reference_features, reference_labels


def encode_labels(labels):
    """
    Encode des labels uniques (par ex. noms de fichiers) en entiers.
    Retourne les labels encodés et un dictionnaire pour déchiffrer.
    """
    unique_labels = list(set(labels))  # Identifie les labels uniques
    label_to_int = {
        label: idx for idx, label in enumerate(unique_labels)
    }  # Mapping label -> entier
    int_to_label = {
        idx: label for label, idx in label_to_int.items()
    }  # Mapping inverse
    encoded_labels = [
        label_to_int[label] for label in labels
    ]  # Convertit les labels en entiers
    return np.array(encoded_labels), label_to_int, int_to_label


def get_caracteristics(nom, images):
    """
    Recupère toutes les caractéristiques des ROIs et des Datasets
    """
    rois = images
    reference_directory = nom
    X_rois, X_references, labels_references = process_data(rois, reference_directory)
    return X_rois, X_references, labels_references


if __name__ == "__main__":
    # Exemple d'utilisation
    # ROIs : tableau contenant les régions d'intérêt (exemple : extrait depuis un document)
    # Répertoire contenant les images de référence
    reference_directory = "data/caracteres"
    nom = f"{reference_directory}/page.png"
    image = cv2.imread(nom, cv2.IMREAD_GRAYSCALE)
    binary_image = Tools.Gaussian_Threshold(image)
    _, rois = Tools.Components_detection(image, binary_image, nom)

    # Extraction et normalisation des données
    X_rois, X_references, labels_references = process_data(rois, reference_directory)

    print(f"Caractéristiques des ROIs : {X_rois.shape}")
    print(f"Caractéristiques des images de référence : {X_references.shape}")
    print(f"Labels des références : {labels_references}")
