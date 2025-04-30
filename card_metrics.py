import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# Paths
MODELS_FOLDER = r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\card_id_models\151_images"
DATA_FOLDER = r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\151_data"
SAVE_FOLDER = r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\card_id_model_more_metrics"

# Helper function
def load_and_preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Main script
os.makedirs(SAVE_FOLDER, exist_ok=True)
model_files = [f for f in os.listdir(MODELS_FOLDER) if f.endswith('.h5')]

for model_file in model_files:
    set_name = model_file.replace('_model.h5', '').replace('.h5', '')
    model_path = os.path.join(MODELS_FOLDER, model_file)
    csv_path = os.path.join(DATA_FOLDER, f"{set_name}_cards.csv")

    if not os.path.exists(csv_path):
        print(f"CSV for {set_name} not found. Skipping...")
        continue

    print(f"\n\nProcessing Model (LOOCV Evaluation): {set_name}")

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Load CSV
    df = pd.read_csv(csv_path)
    image_paths = df.iloc[:, 4].values
    card_ids = df.iloc[:, 0].values

    # Create mapping like during training
    card_id_to_index = {card_id: idx for idx, card_id in enumerate(card_ids)}
    index_to_card_id = {idx: card_id for card_id, idx in card_id_to_index.items()}

    correct = 0
    incorrect = 0

    # Predict one image at a time
    for img_path, true_card_id in zip(image_paths, card_ids):
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}. Skipping...")
            continue
        
        img_array = load_and_preprocess_image(img_path)
        prediction = model.predict(img_array, verbose=0)
        predicted_index = np.argmax(prediction, axis=1)[0]

        predicted_card_id = index_to_card_id.get(predicted_index, "Unknown")

        if predicted_card_id == true_card_id:
            correct += 1
        else:
            incorrect += 1

    total = correct + incorrect
    accuracy = (correct / total) * 100

    print(f"Correct Predictions: {correct}")
    print(f"Incorrect Predictions: {incorrect}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Save folder for this set
    set_save_folder = os.path.join(SAVE_FOLDER, f"{set_name}_metrics")
    os.makedirs(set_save_folder, exist_ok=True)

    # Save metrics text file
    with open(os.path.join(set_save_folder, 'metrics.txt'), 'w') as f:
        f.write(f"Correct Predictions: {correct}\n")
        f.write(f"Incorrect Predictions: {incorrect}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")

    print(f"Metrics saved for {set_name}!\n")

print("All Card ID models processed!")
