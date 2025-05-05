# 2_entrenar_autoencoder.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Cargar datos preprocesados
data = np.load("data/prec_scaled.npy")  # (T, Y, X)

# Usar el 80% para entrenamiento
split = int(0.8 * len(data))
train_data = data[:split]

# Añadir dimensión de canal
train_data = train_data[..., np.newaxis]

# Autoencoder CNN
input_shape = train_data.shape[1:]  # (Y, X, 1)

def build_autoencoder():
    input_img = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2,2), padding='same')(x)
    x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2,2), padding='same')(x)

    x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)
    decoded = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    encoder = models.Model(input_img, encoded)

    return autoencoder, encoder

autoencoder, encoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(train_data, train_data, epochs=50, batch_size=16, shuffle=True)

# Guardar modelos
os.makedirs("modelos", exist_ok=True)
encoder.save("modelos/encoder.h5")
autoencoder.save("modelos/decoder.h5")  # el decoder está implícito en el autoencoder completo

print("Autoencoder entrenado y modelos guardados.")
