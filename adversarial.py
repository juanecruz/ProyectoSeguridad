import tensorflow as tf
import numpy as np

class AdversarialAttack:
    def __init__(self, model):
        self.model = model
    
    def generate_fgsm(self, image, true_label, epsilon=0.1):
        """
        Genera un ataque FGSM (Fast Gradient Sign Method)
        
        Args:
            image: Imagen de entrada (normalizada)
            true_label: Etiqueta verdadera (one-hot encoded)
            epsilon: Magnitud del ataque
            
        Returns:
            Imagen perturbada
        """
        image_tensor = tf.convert_to_tensor(image[np.newaxis, ...])
        true_label = tf.convert_to_tensor(true_label[np.newaxis, ...])
        
        with tf.GradientTape() as tape:
            tape.watch(image_tensor)
            prediction = self.model(image_tensor)
            loss = tf.keras.losses.categorical_crossentropy(true_label, prediction)
            
        gradient = tape.gradient(loss, image_tensor)
        perturbation = epsilon * tf.sign(gradient)
        adversarial_image = image_tensor + perturbation
        adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
        
        return adversarial_image.numpy()[0]