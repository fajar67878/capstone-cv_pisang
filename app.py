import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import cv2
import numpy as np

# Konfigurasi TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

MODEL_PATH = "baseline_banana_model.keras"

# Memuat model
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['Kematangan_Pas', 'Mentah', 'Terlalu_Matang']

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image_bytes = file.read()
        processed_img = preprocess_image(image_bytes)
        predictions = model.predict(processed_img)
        
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0])) * 100

        return jsonify({
            'class': predicted_class,
            'confidence': f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Vercel memerlukan WSGI handler bernama 'app'
if __name__ == '__main__':
    app.run()