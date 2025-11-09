from sklearn.neighbors import NearestNeighbors
import numpy as np

# Datos de ejemplo: matriz de usuarios y ítems
user_item_matrix = [[1, 0, 1, 1],
 [0, 1, 1, 0],
 [1, 1, 0, 1],
 [0, 1, 0, 1]]

# Transponer la matriz para convertir el filtrado basado en usuario a filtrado basado en ítem
item_user_matrix = np.transpose(user_item_matrix)

# Crear un modelo de vecinos más cercanos (kNN)
k = 2 # Número de vecinos a considerar
model = NearestNeighbors(n_neighbors=k, metric='cosine')
model.fit(item_user_matrix)

# Ejemplo de ítem para generar recomendaciones
item_index = 0
item = item_user_matrix[item_index]

# Encontrar los vecinos más cercanos del ítem
distances, indices = model.kneighbors([item])

print(f"distances {distances}")
print(f"dist flattened {distances.flatten()}")

# Recomendaciones basadas en los vecinos más cercanos
recommendations = []
for index in indices.flatten():
    if index != item_index:
        recommendations.append(item_user_matrix[index])

print("Recomendaciones para el ítem:", recommendations)
