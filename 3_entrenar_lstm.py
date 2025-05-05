# 3_entrenar_lstm.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model

# Cargar encoder y datos
encoder = load_model("modelos/encoder.h5")
data = np.load("data/prec_scaled.npy")[..., np.newaxis]  # (T, Y, X, 1)

# Crear embeddings
embeddings = encoder.predict(data)  # (T, h, w, c)
T, h, w, c = embeddings.shape
embeddings_flat = embeddings.reshape(T, -1)  # (T, features)

# Crear secuencias para LSTM
timesteps = 6
X_seq, y_seq = [], []

for i in range(len(embeddings_flat) - timesteps):
    X_seq.append(embeddings_flat[i:i+timesteps])
    y_seq.append(embeddings_flat[i+timesteps])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# Entrenar LSTM
model = models.Sequential([
    layers.LSTM(128, activation='relu', input_shape=(timesteps, X_seq.shape[2])),
    layers.Dense(y_seq.shape[1])
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_seq, y_seq, epochs=50, batch_size=16)

# Guardar modelo
model.save("modelos/lstm.h5")
print("LSTM entrenada y guardada.")
