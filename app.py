import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

# Import TFLite dengan fallback untuk Windows (Localhost) & Netlify (Linux)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

app = Flask(__name__)

# Muat Model TFLite
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.tflite")

def load_model():
    if os.path.exists(MODEL_PATH):
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

@app.route('/')
def home():
    return render_template('index.html') if os.path.exists('templates/index.html') else "Aplikasi Flask Capstone Berjalan!"

@app.route('/predict', methods=['POST'])
def predict():
    if not interpreter:
        return jsonify({'error': 'Model TFLite tidak ditemukan!'}), 500
    
    # Tambahkan logika prediksi kamu di sini
    return jsonify({'status': 'success', 'message': 'Model siap digunakan'})

if __name__ == '__main__':
    app.run(debug=True)