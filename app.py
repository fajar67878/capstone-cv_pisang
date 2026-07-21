import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Memuat model TFLite via tensorflow.lite
MODEL_PATH = os.path.join(os.path.dirname(__file__), "banana_model.tflite")

def get_interpreter():
    if os.path.exists(MODEL_PATH):
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = get_interpreter()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not interpreter:
        return jsonify({'error': 'Model TFLite tidak ditemukan!'}), 500
    
    # Tambahkan logika klasifikasi gambar kamu di sini
    return jsonify({'status': 'success', 'message': 'Model siap memproses gambar'})

if __name__ == '__main__':
    # Membaca port dinamis untuk Replit / Server Cloud
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)