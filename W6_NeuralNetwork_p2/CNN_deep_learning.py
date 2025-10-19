""" partimos de deep_learning.py e implementamos CNN"""
import tensorflow as tf

#importar dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
#cargar dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#normalizar valores de pixeles (el valor maximo es 255 entonces divido por 255)
train_images /= 255.0
test_images /= 255.0

#desarrollar red neuronal
model = tf.keras.models.Sequential([
    #antes de las capas de red neuronal tradicional agregamos capas CONVOLUCIONALES y POOLING
    #capa de 64 filtros 3x3, tamanio de entrada 28x28 con un solo canal de color
    tf.keras.layers.Conv2D(64,(3,3), activation='relu', input_shape=(28,28,1)),
    #reducir imagen con pooling (matriz 2x2)
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    #ahora si layers tradicionales
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') 
])

#Flatten significa que convierte la matrix 28x28 en un vector
#Dense significa que cada neurona esta conectada a todas las neuronas de la capa anterior

model.compile(optimizer=tf.optimizer.Adam(), #ajusta los pesos de red automaticamente
              loss='sparse_categorical_crossentropy', #mide el error del modelo
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

model.evaluate(test_images, test_labels)

#analizar el modelo. Muestra capas, parametros, etc
model.summary()