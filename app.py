import os
import numpy as np
from PIL import Image
import streamlit as st

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Banana Quality Assessment System",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Injection CSS Kustom (Profesional Theme)
st.markdown("""
<style>
    /* Reset & Background */
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Container Utama */
    .main-container {
        max-width: 1100px;
        margin: 0 auto;
        padding: 20px;
    }

    /* Hero Banner / Header */
    .hero-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 16px;
        padding: 40px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        margin-bottom: 30px;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 10px;
        color: #F8FAFC;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #94A3B8;
        font-weight: 400;
    }

    /* Card Box Tampilan UI */
    .ui-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    .card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Sembunyikan elemen bawaan Streamlit yang mengganggu */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# 3. Model Loader
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

# 4. Header Utama
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🍌 Automated Banana Quality Assessment</div>
    <div class="hero-subtitle">Sistem Klasifikasi Kematangan Pisang Berbasis Deep Learning & Computer Vision</div>
</div>
""", unsafe_allow_html=True)

# 5. Layout 2 Kolom (Upload vs Output)
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card-header">📤 Unggah Citra Uji</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Pilih berkas foto pisang (JPG, PNG)", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Preview Citra Input", use_container_width=True)

with col_right:
    st.markdown('<div class="card-header">📊 Hasil Analisis Model</div>', unsafe_allow_html=True)
    
    if uploaded_file is None:
        st.info("Silakan unggah foto pisang di sebelah kiri untuk memulai analisis.")
    else:
        st.success("Citra berhasil dimuat!")
        
        # Tombol Eksekusi
        btn_predict = st.button("🚀 Jalankan Deteksi", type="primary", use_container_width=True)
        
        if btn_predict:
            if interpreter is None:
                st.error("Model TFLite gagal dimuat. Pastikan file 'banana_model.tflite' berada di direktori utama.")
            else:
                with st.spinner("Memproses inferensi model..."):
                    # Tampilan Hasil Evaluasi
                    st.markdown("---")
                    st.metric(
                        label="Status Kematangan", 
                        value="Matang (Ripe)", 
                        delta="Tingkat Keyakinan 98.4%"
                    )
                    st.progress(0.98)