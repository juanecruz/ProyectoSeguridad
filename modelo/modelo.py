import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Carga de datos
train_data = pd.read_csv('/home/juanesteban/Documentos/UltraSecreto/proyecto/modelo/Train.csv')
print("Shape of train_data:", train_data.shape)

# Preprocesamiento
X = train_data.iloc[:, 1:].values
y = train_data.iloc[:, 0].values

# Verificación de dimensiones
assert X.shape[1] == 784, f"Se esperaban 784 features, pero se obtuvieron {X.shape[1]}"
X = X.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y = to_categorical(y, num_classes=10)

# División de datos
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo CNN mejorado
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento con early stopping
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[EarlyStopping(patience=3)])

# Visualización
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Evaluación
test_loss, test_acc = model.evaluate(X_val, y_val, verbose=2)
print(f'\nPrecisión en validación: {test_acc:.2%}')

model.save('../mi_modelo.h5')