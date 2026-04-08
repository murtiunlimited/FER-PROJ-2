# src/models/model.py

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (48, 48)
NUM_CLASSES = 7  # Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

def build_light_model():
    """
    Lightweight CNN model for emotion detection
    """
    inputs = tf.keras.Input(shape=(48, 48, 1))

    # Block 1
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2,2))(x)

    # Block 2
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Block 3
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Fully connected
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model