import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    """Crée un modèle CNN pour classifier les lettres."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(26, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, train_data, train_labels, epochs=10):
    """Entraîne le modèle sur les données."""
    model.fit(train_data, train_labels, epochs=epochs, validation_split=0.2)
    return model

def save_model(model, path="letter_model.h5"):
    """Sauvegarde le modèle."""
    model.save(path)

def load_model(path="letter_model.h5"):
    """Charge un modèle sauvegardé."""
    return tf.keras.models.load_model(path)
