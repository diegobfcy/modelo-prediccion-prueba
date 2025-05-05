# 4_predecir.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Cargar modelos
encoder = load_model("modelos/encoder.h5")
decoder = load_model("modelos/decoder.h5")
lstm = load_model("modelos/lstm.h5")

# Cargar datos originales
data = np.load("data/prec_scaled.npy")[..., np.newaxis]
embeddings = encoder.predict(data)
T, h, w, c = embeddings.shape
embeddings_flat = embeddings.reshape(T, -1)

# Predecir siguiente embedding
timesteps = 6
input_seq = embeddings_flat[-timesteps:]
input_seq = np.expand_dims(input_seq, axis=0)  # (1, 6, features)
predicted_embedding = lstm.predict(input_seq)[0]  # (features,)

# Reconstruir mapa
predicted_embedding_img = predicted_embedding.reshape(h, w, c)
predicted_prec = decoder.predict(np.expand_dims(predicted_embedding_img, axis=0))[0, ..., 0]

# Visualizar
plt.imshow(predicted_prec, cmap='Blues')
plt.colorbar(label="Precipitación [mm/month]")
plt.title("Predicción del mes siguiente")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.show()
