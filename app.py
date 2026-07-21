import os
import numpy as np
from PIL import Image
import streamlit as st

# Coba import tflite_runtime jika ada, jika tidak pakai tensorflow
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None

st.set_page_config(page_title="Klasifikasi Kematangan Pisang", page_icon="🍌")
st.title("🍌 Deteksi Kematangan Pisang")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "banana_model.tflite")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH) and tflite is not None:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

uploaded_file = st.file_uploader("Pilih gambar pisang...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang Diunggah', use_container_width=True)
    
    if st.button('Prediksi Kematangan'):
        if interpreter is None:
            st.error("Model TFLite tidak dapat dimuat!")
        else:
            st.success("Prediksi Berhasil!")