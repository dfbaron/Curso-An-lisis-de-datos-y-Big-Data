import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

# Load and preprocess the dataset
(X_train_val, y_train_val), (X_test, y_test) = fashion_mnist.load_data()

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42)

X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

print(f'El tamaño del dataset de entrenamiento es : {X_train.shape}')
print(f'El tamaño del dataset de Crosvalidación es : {X_val.shape}')
print(f'El tamaño del dataset de prueba es : {X_test.shape}')

print(f'Las categorias a predecir son: {len(np.unique(y_train))}')

# Definir la arquitectura de la red neuronal
model = Sequential([
    Flatten(input_shape=(28, 28), name='Capa_Entrada'),  # Aplanar las imágenes para obtener un arreglo unidimensional
    Dense(128, activation='relu', name='Capa_Oculta_1'),
    Dense(64, activation='relu', name='Capa_Oculta_2'),
    Dense(10, activation='softmax', name='Capa_Salida')],
    name="ModeloClasificacion"
)

# Imprimir el resumen del modelo
print(model.summary())

# Crear un callback para pausar el entrenamiento apenas empeore el Loss de Validación (Evitar overfitting)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluar el modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

y_pred = model.predict(X_test).argmax(axis=1)
p, r, f1, s = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print(f'Test Precision: {p}')
print(f'Test Recall: {r}')
print(f'Test F1-Score: {f1}')

# Guardar el modelo entrenado
model.save('model_trained')
