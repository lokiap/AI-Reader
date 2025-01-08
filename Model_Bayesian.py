import numpy as np
import tensorflow as tf
import Extract as ext
import Analyse
import Tools


def train_naive_bayes(X, y):
    """
    Entraîne un modèle naïf bayésien avec un dataset de référence (une seule observation par classe).
    Retourne les paramètres de la distribution (moyennes, variance globale minimale).
    """
    # Obtenir les classes uniques
    classes = tf.unique(y).y

    # Calcul des priors (P(C))
    priors = {
        int(cls.numpy()): tf.reduce_mean(tf.cast(y == cls, tf.float64))
        for cls in classes
    }

    # Moyennes et variances
    means = {}
    variances = {}

    # Calcul de la variance globale minimale (optionnel)
    global_variance = tf.math.reduce_variance(X, axis=0)
    global_variance = tf.maximum(global_variance, 1e-6)  # Minimum global fixé

    for cls in classes:
        # Applatir les indices
        subset_indices = tf.squeeze(
            tf.where(y == cls)
        )  # Applatir pour obtenir un vecteur d'indices
        subset = tf.gather(X, subset_indices, axis=0)  # Sélectionner les lignes de X

        # Moyennes calculées directement
        means[int(cls.numpy())] = tf.reduce_mean(subset, axis=0)

        # Utilisation de la variance globale minimale
        variances[int(cls.numpy())] = global_variance

    return priors, means, variances


def predict_naive_bayes(X, priors, means, variances):
    """
    Prédictions avec le modèle bayésien, renvoie aussi les probabilités des prédictions.
    """
    predictions = []
    probabilities = []
    classes = list(priors.keys())

    for x in X:
        class_probabilities = {}

        for cls in classes:
            # S'assurer que les variances et priors sont au format float32
            variance = tf.cast(variances[cls], tf.float64)
            prior = tf.cast(priors[cls], tf.float64)

            # Calcul de la probabilité logarithmique P(x|C)
            log_prob = tf.reduce_sum(
                -0.5 * tf.math.log(2 * np.pi * variance)
                - ((x - means[cls]) ** 2) / (2 * variance)
            ) + tf.math.log(prior)
            class_probabilities[cls] = log_prob

        # Extraire les log-probabilités et les transformer en tensor
        log_probs = tf.stack(list(class_probabilities.values()))

        # Normalisation des log-probabilités
        log_probs -= tf.reduce_max(log_probs)  # Pour stabilité numérique
        probs = tf.exp(log_probs)  # Convertir les log-probabilités en probabilités
        probs /= tf.reduce_sum(
            probs
        )  # Normaliser pour que la somme des probabilités soit 1

        # Classe avec la probabilité maximale
        best_class_idx = tf.argmax(
            probs
        )  # On utilise ici `probs` pour trouver l'indice
        best_class = classes[
            best_class_idx.numpy()
        ]  # Récupérer la classe correspondant à l'indice
        predictions.append(best_class)
        probabilities.append(probs[best_class_idx])

    return predictions, probabilities


def standardize(X):
    """Standardiser les caractéristiques en soustrayant la moyenne et en divisant par l'écart-type."""
    mean = tf.reduce_mean(X, axis=0)
    std = tf.math.reduce_std(X, axis=0)
    std = tf.where(std == 0, 1e-6, std)  # Éviter la division par zéro
    return (X - mean) / std


def pipeline(rois, reference_directory):
    """
    Pipeline complet pour entraîner un modèle Bayésien et prédire les labels pour les ROIs.
    """
    # 1. Extraction des caractéristiques
    X_rois = ext.extract_features_from_rois(rois)
    X_references, labels_references_raw = ext.process_reference_images(
        reference_directory
    )

    # 2. Standardisation des caractéristiques
    X_references = standardize(X_references)
    X_rois = standardize(X_rois)

    # 3. Encodage des labels
    labels_references, label_encoder, label_decoder = ext.encode_labels(
        labels_references_raw
    )

    # 4. Entraînement du modèle
    priors, means, variances = train_naive_bayes(X_references, labels_references)

    # 5. Prédictions sur les ROIs avec probabilités
    predicted_labels, probabilities = predict_naive_bayes(
        X_rois, priors, means, variances
    )

    # 6. Décodage des labels prédits
    decoded_labels = [
        label_decoder[label] for label in predicted_labels
    ]  # Pas besoin de .numpy()

    valuable_rois = []
    rois_labels = []
    # 7. Affichage des résultats
    print("Résultats des prédictions :")
    for i, (label, prob) in enumerate(zip(decoded_labels, probabilities)):
        if prob < 0.7:
            continue
        print(f"Composant {i}: {label} (Précision estimée : {prob:.2%})")
        valuable_rois.append(rois[i])
        rois_labels.append(label)
    return valuable_rois, rois_labels


if __name__ == "__main__":
    nom, rois = Analyse.start()
    reference_directory = ""
    if nom.find("page") == -1:
        reference_directory = "data/plan/catalogue"
    else:
        reference_directory = "data/caracteres/catalogue"
    pipeline(rois, reference_directory)
