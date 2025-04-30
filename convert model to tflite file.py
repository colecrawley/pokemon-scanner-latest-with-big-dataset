import tensorflow as tf
import os
import numpy as np
from keras.models import load_model

# Utility: get file size
def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size

# Utility: convert size to readable format
def convert_bytes(size, unit=None):
    if unit == "KB":
        print('File size:', round(size / 1024, 3), 'Kilobytes')
    elif unit == "MB":
        print('File size:', round(size / (1024 * 1024), 3), 'Megabytes')
    else:
        print('File size:', size, 'bytes')

# Paths for the set model
model_path = r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\dataset_for_model\setModel.h5"
tflite_path = r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\dataset_for_model\setModel.tflite"

print(f"\nConverting Set Model: {os.path.basename(model_path)}")

# Load the model
model = load_model(model_path)
print("Original model input shape:", model.input_shape)

# Set input shape (assuming 224x224x3 input images)
input_shape = (1, 224, 224, 3)
dummy_input = tf.ones(input_shape, dtype=tf.float32)
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec(input_shape, tf.float32))

# Convert to TFLite using the concrete function
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

# Save the TFLite model
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

# Report file size
convert_bytes(get_file_size(tflite_path), "KB")

# Load and inspect the TFLite model
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Shape:", input_details[0]['shape'])
print("Input Type:", input_details[0]['dtype'])
print("Output Shape:", output_details[0]['shape'])
print("Output Type:", output_details[0]['dtype'])
