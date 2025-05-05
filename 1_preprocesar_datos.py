# 1_preprocesar_datos.py
import xarray as xr
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler

# Cargar archivo NetCDF
ds = xr.open_dataset("data/data.nc")
prec = ds['Prec'].values  # (T, Y, X)

# Normalizar datos entre 0 y 1
scaler = MinMaxScaler()
T, Y, X = prec.shape
prec_reshaped = prec.reshape(T, Y * X)
prec_scaled = scaler.fit_transform(prec_reshaped)
prec_scaled = prec_scaled.reshape(T, Y, X)

# Guardar datos escalados
os.makedirs("data", exist_ok=True)
np.save("data/prec_scaled.npy", prec_scaled)

print("Datos preprocesados y guardados en data/prec_scaled.npy")
