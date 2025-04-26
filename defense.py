import tensorflow as tf
import numpy as np

def adversarial_training(model, X_train, y_train, epochs=3, batch_size=32, epsilon=0.1):
    """Entrena el modelo con ejemplos adversariales"""
    attack = AdversarialAttack(model)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            # Generar ejemplos adversariales
            X_adv = np.array([
                attack.generate_fgsm(img, label, epsilon)[0] 
                for img, label in zip(X_batch, y_batch)
            ])
            
            # Entrenamiento combinado
            model.fit(
                np.concatenate([X_batch, X_adv]),
                np.concatenate([y_batch, y_batch]),
                batch_size=batch_size,
                verbose=0
            )
    
    model.save('ProyectoSeguridad/modelo/modelo_robusto.h5')
    return model