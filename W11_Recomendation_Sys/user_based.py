from sklearn.neighbors import NearestNeighbors
# Datos de ejemplo: matriz de usuarios y ítems
user_item_matrix = [[1, 0, 1, 1],
 [0, 1, 1, 0],
 [1, 1, 0, 1],
 [0, 1, 0, 1]]

# Crear un modelo de vecinos más cercanos (kNN)
k = 2 # Número de vecinos a considerar
model = NearestNeighbors(n_neighbors=k, metric='cosine') #metrica de coseno
model.fit(user_item_matrix)

# Ejemplo de usuario para generar recomendaciones
user_index = 0 #usuario ejemplo
user = user_item_matrix[user_index]

# Encontrar los vecinos más cercanos del usuario
distances, indices = model.kneighbors([user])

# Recomendaciones basadas en los vecinos más cercanos
recommendations = []
for index in indices.flatten():
    if index != user_index:
        recommendations.append(user_item_matrix[index])
print("Recomendaciones para el usuario:", recommendations)