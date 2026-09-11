import tensorflow as tf

# Load your existing H5 model
model = tf.keras.models.load_model("cat_dog_cnn_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # reduces file size & RAM
tflite_model = converter.convert()

# Save the new lightweight model
with open("cat_dog_cnn_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved cat_dog_cnn_model.tflite successfully!")