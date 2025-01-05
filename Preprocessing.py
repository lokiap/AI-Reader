import cv2
import numpy as np

def preprocess_image(img, size=(28, 28)):
    """Redimensionne et normalise une image."""
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img_resized / 255.0

def prepare_training_data(components, labels):
    """
    Prépare les données d'entraînement à partir des composants extraits et de leurs étiquettes.
    :param components: Liste des ROI extraites [(roi, (x, y, w, h))].
    :param labels: Liste des étiquettes correspondantes (A=0, B=1, ... Z=25).
    :return: train_data, train_labels.
    """
    train_data = []
    train_labels = []

    for i, (roi, _) in enumerate(components):
        processed_roi = preprocess_image(roi)
        train_data.append(processed_roi)
        train_labels.append(labels[i])

    train_data = np.array(train_data).reshape(-1, 28, 28, 1)
    train_labels = np.array(train_labels)
    return train_data, train_labels

def label_components_interactively(components):
    """
    Demande à l'utilisateur de saisir une étiquette pour chaque composant.
    :param components: Liste des ROI [(roi, (x, y, w, h))].
    :return: Liste des étiquettes.
    """
    labels = []
    for i, (roi, _) in enumerate(components):
        cv2.imshow(f"Composant {i+1}", roi)
        label = input(f"Entrez l'étiquette pour le composant {i+1} (A-Z) : ").upper()
        labels.append(ord(label) - ord('A'))
        cv2.destroyAllWindows()
    return labels
