import os
import numpy as np
from PIL import Image
import streamlit as st

# Konfigurasi Halaman & Theme
st.set_page_config(
    page_title="Sistem Deteksi Kematangan Pisang",
    page_icon="🍌",
    layout="centered"
)

# Custom CSS untuk Ubah Total Tampilan Streamlit
st.markdown("""
    <style>
    /* Ubah Background & Font Utama */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header Card Custom */
    .header-card {
        background: linear-gradient(135deg, #ffe100 0%, #ff9900 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: #2b2b2b;
        text-align: center;
        box-shadow: 0 10px 25px rgba(255, 153, 0, 0.2);
        margin-bottom: 2rem;
    }
    
    .header-card h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
    }
    
    .header-card p {
        margin-top: 0.5rem;
        font-size: 1.05rem;
        opacity: 0.9;
    }

    /* Container untuk Upload & Output */
    .content-box {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }

    /* Sembunyikan Header / Footer Default Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 1. Tampilan Header Utama
st.markdown("""
    <div class="header-card">
        <h1>🍌 Detection System</h1>
        <p>Klasifikasi Kematangan Pisang Berbasis Artificial Intelligence</p>
    </div>
""", unsafe_allow_html=True)

# Load Model TFLite
MODEL_PATH = os.path.join(os.path.dirname(__file__), "banana_model.tflite")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        except ImportError:
            import tensorflow.lite as tflite
            interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    return None

interpreter = load_model()

# 2. Section Input / Upload
st.subheader("📸 Unggah Citra Pisang")
uploaded_file = st.file_uploader("Format yang didukung: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Preview Gambar:**")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("**Analisis Kematangan:**")
        st.write("Klik tombol di bawah untuk memproses gambar menggunakan model TFLite.")
        
        if st.button('🚀 Mulai Prediksi', type="primary", use_container_width=True):
            if interpreter is None:
                st.error("Model TFLite belum terdeteksi/gagal dimuat.")
            else:
                with st.spinner('Menganalisis citra...'):
                    # LOGIKA MODEL KAMU DI SINI
                    st.success("✅ Prediksi Selesai!")
                    st.metric(label="Hasil Klasifikasi", value="Matang (Ripe)", delta="Confidence: 98.5%")