from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('ProyectoSeguridad/mi_modelo.h5') 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        image = image.convert('L').resize((28, 28))
        image_array = np.array(image)
        image_array = 255 - image_array
        image_array = image_array.reshape(1, 28, 28, 1).astype('float32') / 255.0
        
        prediction = model.predict(image_array)
        predicted_num = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return jsonify({
            'prediction': int(predicted_num),
            'confidence': float(confidence),
            'probabilities': prediction[0].tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/attack', methods=['POST'])
def attack():
    try:
        import tensorflow as tf
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        image = image.convert('L').resize((28, 28))
        image_array = np.array(image)
        image_array = 255 - image_array
        image_array = image_array.reshape(1, 28, 28, 1).astype('float32') / 255.0

        dummy_label = model.predict(image_array)
        dummy_label = tf.convert_to_tensor(dummy_label)

        image_tensor = tf.convert_to_tensor(image_array)
        image_tensor = tf.cast(image_tensor, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(image_tensor)
            prediction = model(image_tensor)
            loss = tf.keras.losses.categorical_crossentropy(dummy_label, prediction)
        gradient = tape.gradient(loss, image_tensor)
        signed_grad = tf.sign(gradient)
        epsilon = 0.1
        adversarial_image = image_tensor + epsilon * signed_grad
        adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)

        new_prediction = model.predict(adversarial_image)
        predicted_num = int(np.argmax(new_prediction))
        confidence = float(np.max(new_prediction))

        adv_np = adversarial_image.numpy().reshape(28, 28) * 255
        adv_img_pil = Image.fromarray(adv_np.astype(np.uint8), mode='L')
        buffered = io.BytesIO()
        adv_img_pil.save(buffered, format="PNG")
        adv_base64 = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'prediction': predicted_num,
            'confidence': confidence,
            'adversarial_image': 'data:image/png;base64,' + adv_base64
        })
        
    except Exception as e:
        print(f"[ERROR]: {str(e)}")
        return jsonify({'error': 'Internal error in adversarial attack'}), 500


@app.route('/defend', methods=['POST'])
def defend():
    try:
        import cv2
        import tensorflow as tf
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        image = image.convert('L').resize((28, 28))
        image_array = np.array(image)
        image_array = 255 - image_array

        image_array = cv2.GaussianBlur(image_array, (3, 3), 0)

        image_array = image_array.reshape(1, 28, 28, 1).astype('float32') / 255.0

        prediction = model.predict(image_array)
        predicted_num = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        img_uint8 = (image_array.reshape(28, 28) * 255).astype(np.uint8)
        cleaned_pil = Image.fromarray(img_uint8, mode='L')
        buffered = io.BytesIO()
        cleaned_pil.save(buffered, format="PNG")
        cleaned_base64 = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'prediction': predicted_num,
            'confidence': confidence,
            'cleaned_image': 'data:image/png;base64,' + cleaned_base64
        })
    except Exception as e:
        print(f"[DEFENSE ERROR]: {str(e)}")
        return jsonify({'error': 'Internal error in defense'}), 500
    

if __name__ == '__main__':
    app.run(debug=True)