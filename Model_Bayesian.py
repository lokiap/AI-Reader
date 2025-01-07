import numpy as np
import tensorflow as tf
import Extract as ext
import Tools


def train_naive_bayes(X, y):
    """
    Entraîne un modèle naïf bayésien avec un dataset de référence (une seule observation par classe).
    Retourne les paramètres de la distribution (moyennes, variance globale minimale).
    """
    # Obtenir les classes uniques
    classes = tf.unique(y).y

    # Calcul des priors (P(C))
    priors = {cls: tf.reduce_mean(tf.cast(y == cls, tf.float32)) for cls in classes}

    # Moyennes et variances
    means = {}
    variances = {}

    # Calcul de la variance globale minimale (optionnel)
    global_variance = tf.math.reduce_variance(X, axis=0)
    global_variance = tf.maximum(global_variance, 1e-6)  # Minimum global fixé

    for cls in classes:
        subset = X[tf.where(y == cls)[:, 0]]

        # Moyennes calculées directement
        means[cls] = tf.reduce_mean(subset, axis=0)

        # Utilisation de la variance globale minimale
        variances[cls] = global_variance
        print(f"Classe {cls} :")
        print(f"  Moyenne : {means[cls]}")
        print(f"  Variance (globale) : {variances[cls]}")  # Debugging

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
            # Calcul de la probabilité logarithmique P(x|C)
            log_prob = tf.reduce_sum(
                -0.5 * tf.log(2 * np.pi * variances[cls])
                - ((x - means[cls]) ** 2) / (2 * variances[cls])
            ) + tf.log(priors[cls])
            class_probabilities[cls] = log_prob

        # Debug : Affichage des log-probabilités avant normalisation
        print("Log-probabilités avant normalisation :")
        for cls, log_prob in class_probabilities.items():
            print(f"Classe: {cls}, Log-probabilité: {log_prob}")

        # Normalisation des log-probabilités
        log_probs = tf.stack(list(class_probabilities.values()))
        log_probs -= tf.reduce_max(log_probs)  # Pour stabilité numérique
        probs = tf.exp(log_probs)  # Convertir les log-probabilités en probabilités
        probs /= tf.reduce_sum(
            probs
        )  # Normaliser pour que la somme des probabilités soit 1

        # Classe avec la probabilité maximale
        best_class = tf.argmax(class_probabilities, axis=0)
        predictions.append(best_class)
        probabilities.append(probs[best_class])

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
    decoded_labels = [label_decoder[label.numpy()] for label in predicted_labels]

    # 7. Affichage des résultats
    print("Résultats des prédictions :")
    for i, (roi, prob) in enumerate(zip(decoded_labels, probabilities)):
        print(f"Composant {i}: {roi} (Précision estimée : {prob:.2%})")


if __name__ == "__main__":
    image, nom = Tools.Choix()
    input = input("1 - Threshold \n2 - Gaussian \n3 - Mean_Threshold\n")
    binary_image = ""
    if input == "1":
        binary_image = Tools.Threshold(image)
    elif input == "2":
        binary_image = Tools.Gaussian_Threshold(image)
    elif input == "3":
        binary_image = Tools.Mean_Threshold(image)
    _, rois = Tools.Components_detection(image, binary_image, nom)
    reference_directory = ""
    if nom.find("page") == -1:
        reference_directory = "data/plan/catalogue"
    else:
        reference_directory = "data/caracteres/catalogue"
    pipeline(rois, reference_directory)
