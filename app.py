import cv2
import numpy as np
import tensorflow as tf

# Muat model terbaik (misal hasil transfer learning yang biasanya lebih akurat)
try:
    model = tf.keras.models.load_model('mobilenet_banana_model.keras')
    print("Berhasil memuat model Transfer Learning.")
except:
    model = tf.keras.models.load_model('baseline_banana_model.keras')
    print("Berhasil memuat model Baseline.")

# Daftar label sesuai urutan abjad folder kelas Keras
class_labels = ['Overripe (Terlalu Matang)', 'Ripe (Matang Sempura)', 'Rotten (Busuk)', 'Unripe (Mentah)']

# Aktifkan Kamera Laptop
cap = cv2.VideoCapture(0)
print("Aplikasi Deteksi Kematangan Pisang Aktif... Tekan 'q' pada keyboard untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing bingkai gambar dari kamera (150x150)
    resized_frame = cv2.resize(frame, (150, 150))
    img_array = np.expand_dims(resized_frame, axis=0)

    # Lakukan Prediksi
    predictions = model.predict(img_array, verbose=0)[0]
    best_class_index = np.argmax(predictions) # Cari nilai probabilitas tertinggi
    confidence = predictions[best_class_index] * 100

    # Tentukan text label dan warna bounding text
    label_text = f"{class_labels[best_class_index]} ({confidence:.2f}%)"
    
    # Warna teks berdasarkan hasil deteksi
    if best_class_index == 3:   # Mentah
        color = (255, 0, 0)     # Biru
    elif best_class_index == 1: # Matang
        color = (0, 255, 0)     # Hijau
    else:                       # Terlalu matang / Busuk
        color = (0, 0, 255)     # Merah

    # Tampilkan Teks Hasil Prediksi di Layar Video Kamera
    cv2.putText(frame, f"Status: {label_text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow('Capstone Project AI - Kematangan Pisang', frame)

    # Berhenti jika menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()