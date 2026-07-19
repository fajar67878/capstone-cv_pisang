from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Konfigurasi folder penyimpanan gambar sementara
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Memuat model terbaik hasil training
try:
    model = tf.keras.models.load_model('mobilenet_banana_model.keras')
    print("Sistem sukses memuat model Transfer Learning.")
except:
    model = tf.keras.models.load_model('baseline_banana_model.keras')
    print("Sistem sukses memuat model Baseline.")

# Urutan label disesuaikan dengan urutan kelas alfabetis default Keras
class_labels = ['Overripe (Terlalu Matang)', 'Ripe (Matang Sempurna)', 'Rotten (Busuk)', 'Unripe (Mentah)']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction=None)
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction=None)
        
        if file:
            # Simpan file gambar yang diunggah
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # --- 🛠️ PROSES PREPROCESSING CITRA YANG BENAR ---
            # 1. Baca gambar (OpenCV membaca dalam format BGR)
            img = cv2.imread(filepath)
            
            # 2. KONVERSI BGR KE RGB (Agar warna tidak terbalik di mata TensorFlow)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 3. Ubah ukuran sesuai input dimensi model (150x150)
            img_resized = cv2.resize(img_rgb, (150, 150))
            
            # 4. Tambahkan dimensi batch
            img_array = np.expand_dims(img_resized, axis=0)
            
            # --- INFERENSI MODEL AI ---
            predictions = model.predict(img_array, verbose=0)[0]
            best_class_index = np.argmax(predictions)
            
            # Ambil nilai probabilitas tertinggi & format menjadi persentase
            confidence_val = f"{predictions[best_class_index] * 100:.2f}"
            label_val = class_labels[best_class_index]
            
            return render_template('index.html', 
                                   prediction=True, 
                                   label=label_val, 
                                   confidence=confidence_val, 
                                   image_path=filepath)
            
    return render_template('index.html', prediction=None, label=None, confidence=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True, port=5000)