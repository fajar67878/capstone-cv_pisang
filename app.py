import os
import requests

# Konfigurasi agar TensorFlow hemat RAM & menggunakan CPU di Railway
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

from flask import Flask, render_template, request as flask_request, jsonify
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

MODEL_PATH = "baseline_banana_model.keras"
MODEL_URL = "https://github.com/fajar67878/capstone-cv_pisang/releases/download/v1.0/baseline_banana_model.keras"

def download_file(url, target_path):
    print(f"Mengunduh model dari: {url}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
    
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download model SUKSES!")
    else:
        raise Exception(f"Gagal download model. Status Code: {response.status_code}")

# Auto-download model jika belum ada di server
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
    try:
        download_file(MODEL_URL, MODEL_PATH)
    except Exception as e:
        print(f"Error saat mengunduh model: {e}")

# Memuat model Keras
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
    if 'file' not in flask_request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = flask_request.files['file']
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