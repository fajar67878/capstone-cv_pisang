import os
import numpy as np
from PIL import Image
import streamlit as st

# ==========================================
# 1. KONFIGURASI HALAMAN & CUSTOM CSS (UI)
# ==========================================
st.set_page_config(
    page_title="Banana Quality Assessment System",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Background Utama */
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Banner Hero */
    .hero-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 16px;
        padding: 35px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
        color: #F8FAFC;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #94A3B8;
        font-weight: 400;
    }

    /* Judul Bagian Card */
    .card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 16px;
    }

    /* Sembunyikan elemen bawaan Streamlit */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL TFLITE (SAFE IMPORT)
# ==========================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "banana_model.tflite")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
        
    # Coba import bertahap agar aman di berbagai environment Python
    try:
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path=MODEL_PATH)
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
            interpreter = Interpreter(model_path=MODEL_PATH)
        except ImportError:
            try:
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            except ImportError:
                return None

    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# ==========================================
# 3. HEADER UTAMA
# ==========================================
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🍌 Automated Banana Quality Assessment</div>
    <div class="hero-subtitle">Sistem Klasifikasi Kematangan Pisang Berbasis Machine Learning & Computer Vision</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. LAYOUT DUA KOLOM (UPLOAD & HASIL)
# ==========================================
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
        st.success("Citra siap diproses!")
        btn_predict = st.button("🚀 Jalankan Deteksi", type="primary", use_container_width=True)
        
        if btn_predict:
            if interpreter is None:
                st.error("Model `banana_model.tflite` tidak ditemukan atau gagal dimuat.")
            else:
                with st.spinner("Menganalisis gambar..."):
                    try:
                        # 1. Preprocessing Gambar
                        input_details = interpreter.get_input_details()
                        output_details = interpreter.get_output_details()
                        
                        # Ambil ukuran input yang diminta model (biasanya 224x224)
                        input_shape = input_details[0]['shape']
                        img_height, img_width = input_shape[1], input_shape[2]
                        
                        img_resized = image.convert('RGB').resize((img_width, img_height))
                        img_array = np.array(img_resized, dtype=np.float32) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)

                        # 2. Inisialisasi Tipe Data Input
                        if input_details[0]['dtype'] == np.uint8:
                            img_array = (img_array * 255).astype(np.uint8)

                        # 3. Jalankan Prediksi
                        interpreter.set_tensor(input_details[0]['index'], img_array)
                        interpreter.invoke()
                        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

                        # 4. Daftar Label (Sesuaikan urutannya dengan model kamu)
                        labels = ["Belum Matang (Unripe)", "Matang (Ripe)", "Terlalu Matang (Overripe)"]
                        
                        if len(predictions) == len(labels):
                            top_class_idx = int(np.argmax(predictions))
                            confidence = float(predictions[top_class_idx]) * 100
                            result_label = labels[top_class_idx]
                        else:
                            # Fallback jika model outputnya single value
                            confidence = float(predictions[0]) * 100
                            result_label = "Matang (Ripe)" if confidence > 50 else "Belum Matang (Unripe)"

                        # 5. Tampilkan Hasil
                        st.markdown("---")
                        st.metric(
                            label="Prediksi Status Kematangan", 
                            value=result_label, 
                            delta=f"Tingkat Keyakinan: {confidence:.1f}%"
                        )
                        st.progress(min(confidence / 100, 1.0))
                        
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")