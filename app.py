import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Klasifikasi Kematangan Pisang", page_icon="🍌")

st.title("🍌 Deteksi Kematangan Pisang")
st.write("Unggah foto pisang untuk mendeteksi tingkat kematangannya secara otomatis.")

# Load Model TFLite
MODEL_PATH = os.path.join(os.path.dirname(__file__), "banana_model.tflite")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

# Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar pisang...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang Diunggah', use_column_width=True)
    
    if st.button('Prediksi Kematangan'):
        if interpreter is None:
            st.error("Model TFLite tidak ditemukan! Pastikan file banana_model.tflite ada di repo.")
        else:
            with st.spinner('Memproses gambar...'):
                # Logika inferensi model kamu di sini
                st.success("Prediksi Berhasil! (Pisang Matang)")