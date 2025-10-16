# Paso 1: Importar las bibliotecas necesarias
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# Paso 2: Preparar los datos de ejemplo
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11], [11, 12], [12, 10], [6, 7], [7, 6]])

# Paso 3: Crear y ajustar el modelo de DBSCAN
dbscan = DBSCAN(eps=1, min_samples=2) # Especificamos el valor de epsilon y el número mínimo de ejemplos
dbscan.fit(X)

# Paso 4: Obtener las etiquetas de los clusters y los outliers
labels = dbscan.labels_
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True

# Paso 5: Visualizar los resultados
unique_labels = np.unique(labels)
colors = ['r', 'g', 'b', 'y']

for label, color in zip(unique_labels, colors):
    if label == -1: # Outliers
        color = 'k'

    class_member_mask = (labels == label)

    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], color=color, marker='o')

    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], color=color, marker='x')

plt.show()