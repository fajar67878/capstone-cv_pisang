import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling
import matplotlib.pyplot as plt

# 1. Pengaturan Dataset & Preprocessing (Minggu 2)
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Mengarah ke folder yang sesuai di VS Code Anda
train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/train',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/valid',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Ambil nama kelas secara otomatis (overripe, ripe, rotten, unripe)
class_names = train_ds.class_names
print("Kelas yang terdeteksi:", class_names)

# Data Augmentasi untuk variasi data (Syarat Minggu 2)
data_augmentation = Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
])

# 2. Arsitektur Model Baseline Custom (Pendekatan 1)
model = Sequential([
    Rescaling(1./255, input_shape=(150, 150, 3)),
    data_augmentation,
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax') # 4 Output untuk 4 kelas pisang
])

# Kompilasi Model dengan Metrik Evaluasi yang Tepat (Syarat Rubrik)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. Training Model (Minggu 2-3)
epochs = 10
print("Memulai training model baseline...")
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Simpan Model Baseline Berformat Keras Terbaru (.keras atau .h5)
model.save('baseline_banana_model.keras')
print("Model baseline berhasil disimpan!")

# 4. Visualisasi Hasil Evaluasi untuk Laporan (Minggu 3)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Akurasi Model Baseline')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Model Baseline')
plt.legend()

plt.savefig('baseline_evaluation.png')
print("Grafik evaluasi disimpan sebagai 'baseline_evaluation.png'")
plt.show()