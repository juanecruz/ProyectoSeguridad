from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('proyecto/mi_modelo.h5') 

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

if __name__ == '__main__':
    app.run(debug=True)
