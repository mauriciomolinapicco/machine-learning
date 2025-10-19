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
    tf.keras.layers.Flatten(input_shape=(28,28)), #tamano de imagen 28x28
    tf.keras.layers.Dense(128, activation='relu'), #128 neuronas y f de activacion relu
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') #una neurona por clase/categoria
])

#Flatten significa que convierte la matrix 28x28 en un vector
#Dense significa que cada neurona esta conectada a todas las neuronas de la capa anterior

model.compile(optimizer=tf.optimizer.Adam(), #ajusta los pesos de red automaticamente
              loss='sparse_categorical_crossentropy', #mide el error del modelo
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

model.evaluate(test_images, test_labels)