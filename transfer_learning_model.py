import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Rescaling
import matplotlib.pyplot as plt

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Membaca dataset
train_ds = tf.keras.utils.image_dataset_from_directory('dataset/train', image_size=IMG_SIZE, batch_size=BATCH_SIZE)
val_ds = tf.keras.utils.image_dataset_from_directory('dataset/valid', image_size=IMG_SIZE, batch_size=BATCH_SIZE)

# Pendekatan 2: Menggunakan Base Model Pra-latih MobileNetV2 (Transfer Learning)
base_model = tf.keras.applications.MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
base_model.trainable = False # Membekukan bobot awal

# Menyusun Arsitektur
model_tl = Sequential([
    Rescaling(1./255, input_shape=(150, 150, 3)),
    base_model,
    GlobalAveragePooling2D(),
    Dense(4, activation='softmax') # 4 Output untuk 4 kelas pisang
])

model_tl.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

# Training Model Kedua
epochs = 10
print("Memulai training model transfer learning...")
history_tl = model_tl.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Simpan Model Transfer Learning
model_tl.save('mobilenet_banana_model.keras')
print("Model transfer learning berhasil disimpan!")

# Visualisasi Hasil untuk Pembanding di Laporan
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history_tl.history['accuracy'], label='Train Accuracy (TL)')
plt.plot(history_tl.history['val_accuracy'], label='Val Accuracy (TL)')
plt.title('Akurasi Transfer Learning')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_tl.history['loss'], label='Train Loss (TL)')
plt.plot(history_tl.history['val_loss'], label='Val Loss (TL)')
plt.title('Loss Transfer Learning')
plt.legend()

plt.savefig('transfer_learning_evaluation.png')
print("Grafik evaluasi disimpan sebagai 'transfer_learning_evaluation.png'")
plt.show()