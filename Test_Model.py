from Tools import *
from Preprocessing import prepare_training_data ,label_components_interactively
from Model import *

def main():
    # Étape 1 : Charger et prétraiter l'image
    image, nom = Choix()
    binary_image = Gaussian_Threshold(image)
    color_image, components = Components_detection(image, binary_image, nom)

    # Étape 2 : Associer les étiquettes aux ROI
    print("Veuillez étiqueter les composants extraits.")
    labels = label_components_interactively(components)

    # Étape 3 : Préparer les données pour l'entraînement
    train_data, train_labels = prepare_training_data(components, labels)

    # Étape 4 : Créer et entraîner le modèle
    model = create_model()
    model = train_model(model, train_data, train_labels, epochs=10)

    # Étape 5 : Sauvegarder le modèle
    save_model(model)
    print("Modèle entraîné et sauvegardé sous 'letter_model.h5'.")

if __name__ == "__main__":
    main()
