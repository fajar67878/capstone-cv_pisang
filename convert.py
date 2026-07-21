import tf2onnx
import tensorflow as tf

print("Memuat model Keras...")
model = tf.keras.models.load_model("baseline_banana_model.keras")

print("Mengonversi ke format ONNX...")
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
tf2onnx.convert.from_keras(model, input_signature=spec, output_path="banana_model.onnx")

print("BERHASIL! File banana_model.onnx sudah dibuat.")