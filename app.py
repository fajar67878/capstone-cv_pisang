import os
import urllib.request
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

# Config Model & Release Link
MODEL_PATH = "baseline_banana_model.keras"
MODEL_URL = "https://github.com/fajar67878/capstone-cv_pisang/releases/download/v1.0/baseline_banana_model.keras"

# 1. Download model otomatis jika belum ada di server Railway
if not os.path.exists(MODEL_PATH):
    print(f"Downloading model from {MODEL_URL} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download model selesai!")

# 2. Load model TensorFlow/Keras
model = tf.keras.models.load_model(MODEL_PATH)

# Sesuaikan dengan nama kelas/label dataset kamu
CLASS_NAMES = ['Kematangan_Pas', 'Mentah', 'Terlalu_Matang']

def preprocess_image(image_bytes):
    # Convert bytes image ke format OpenCV/NumPy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Preprocessing (Sesuaikan ukuran input model kamu, contoh 224x224 atau 150x150)
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

if __name__ == '__main__':

    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)