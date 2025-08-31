Con Scikit-Learn solamente se necesitan 4 pasos para
entrenar cualquier algoritmo de ML:

1. Importar de la librería Scikit-Learn el módulo que contiene el algoritmo de ML que
queremos utilizar. En este caso, queremos realizar regresión logística, por lo tanto,
importamos el módulo LogisticRegression.
# Paso 1 - Importar modulo
from sklearn.linear_model import LogisticRegression


2. Crear una instancia del modelo.
# Paso 2 - Crear una instancia del modelo
logisticRegr = LogisticRegression()


3. Entrenar el modelo. Para ello utilizamos el método .fit(x_train,y_train), el cual
recibe por parámetro el dataset de entrenamiento y los labels.
# Paso 3 - Entrenar el modelo con el dataset de entrenamiento
logisticRegr.fit(x_train, y_train)


4. Predecir el label para nuevas imágenes. Para ello, utilizamos el método .predict(x),
el cual recibe por parámetro una nueva imagen de la cual se desconoce el label. El
objetivo de este método consiste en determinar dicho label. En este caso
probamos primero con la primera imagen del dataset de testeo.
# Paso 4 - Crear predicciones para imagenes nuevas
logisticRegr.predict(x_test[0].reshape(1,-1))

# Imprimimos la imagen para chequear si la predicción es correcta
plt.figure(figsize=(3,1))
image = x_test[0]
plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)