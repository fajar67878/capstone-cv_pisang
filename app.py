import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

# Handling import tflite untuk Windows & Netlify Linux
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None

app = Flask(__name__)

# Menggunakan model TFLite yang ada di direktori
MODEL_PATH = os.path.join(os.path.dirname(__file__), "banana_model.tflite")

def get_interpreter():
    if os.path.exists(MODEL_PATH) and tflite is not None:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    return None

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)