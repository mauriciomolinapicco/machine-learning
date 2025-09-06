import numpy as np
import pandas as pd

# Fijar semilla para reproducibilidad
np.random.seed(0)

# Cantidad de filas a generar
num_filas = 20

# Generar labels aleatorios entre 0 y 9
labels = np.random.randint(0, 10, num_filas)

# Generar matriz de pixeles aleatorios (0-255) de tamaño num_filas x 784
pixeles = np.random.randint(0, 256, (num_filas, 784))

# Crear DataFrame
df = pd.DataFrame(
    np.column_stack((labels, pixeles)),
    columns=["label"] + [f"pixel{i}" for i in range(784)]
)

# Guardar a CSV
df.to_csv("digits_small.csv", index=False)

print("CSV generado con éxito: digits_small.csv")
print(df.head())