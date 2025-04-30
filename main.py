import os
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Define image path and load categories
imagePath = r"test_images\surfing pikachu wizard.png"  # Adjust test images here
categories = os.listdir(r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\dataset_for_model\train")
categories.sort()
print(categories)

# Load the saved model
path_for_saved_model = r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\dataset_for_model\setModel.h5"
model = load_model(path_for_saved_model)

print(model.summary())

# Define image classification function
def classify_image(image_path):
    # Load the image
    img = Image.open(image_path)

    # ✅ Ensure the image is in RGB (convert if needed)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize the image to the expected size
    img = img.resize((224, 224))

    # Convert to array and preprocess
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Shape: (1, 224, 224, 3)
    x = preprocess_input(x)

    # Predict
    pred = model.predict(x)

    # Top 5 predictions
    top_5_predictions = np.argsort(pred[0])[-5:][::-1]

    return top_5_predictions, pred


# Test the image classification and display top 5 predictions
top_5_predictions, pred = classify_image(imagePath)

# Get class labels
categories = os.listdir(r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\dataset_for_model\train")
categories.sort()

# Print top 5 predictions with their probabilities
for i in top_5_predictions:
    print(f"{categories[i]}: {pred[0][i]:.4f}")

# Read the image
img = cv2.imread(imagePath)

# Check if image is read correctly
if img is None:
    print("Error: Image not found or unable to read.")
else:
    # Add text to the image (display the top 5 predictions)
    resultText = '\n'.join([f"{categories[i]}: {pred[0][i]:.4f}" for i in top_5_predictions])
    img = cv2.putText(img, resultText, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Convert BGR to RGB (matplotlib uses RGB format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide axes
    plt.show()


################################################

# Card ID prediction function (without rarity)

# Function to get card IDs from the CSV file of the predicted set
def get_card_ids_from_csv(set_name):
    # Construct the path to the CSV file for the set
    csv_path = os.path.join(r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\151_data", f"{set_name}_cards.csv")
    
    # Check if the file exists
    if not os.path.exists(csv_path):
        print(f"Error: File not found for set: {set_name}")
        return None
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Get the card IDs (as they are)
    card_ids = df['id'].values  # Assuming 'id' column exists in your CSV
    
    return card_ids

# Function to load card ID model
def load_card_id_model(set_name):
    # Path for the set-specific card ID model (assuming it's stored similarly as the set model)
    card_id_model_path = os.path.join(r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\card_id_models\151_images", f"{set_name}_model.h5")
    
    # Check if the model file exists
    if not os.path.exists(card_id_model_path):
        print(f"Error: Model not found for set: {set_name}")
        return None
    
    card_id_model = tf.keras.models.load_model(card_id_model_path)
    
    return card_id_model

# Function to predict the card ID (without rarity)
def predict_card(imageFile):
    # Predict the set first
    top_5_predictions, pred = classify_image(imageFile)
    
    # Get the predicted set (use the top prediction for the set name)
    predicted_set = categories[top_5_predictions[0]]

    # Load the card IDs for the predicted set from the CSV file
    card_ids = get_card_ids_from_csv(predicted_set)
    if card_ids is None:
        return None, None, None
    
    # Load the card ID model for the predicted set
    card_id_model = load_card_id_model(predicted_set)
    if card_id_model is None:
        return None, None, None
    
    # Preprocess the image for prediction
    img = Image.open(imageFile).convert("RGB")
    img = img.resize((224, 224))
    x = np.expand_dims(image.img_to_array(img), axis=0)
    x = preprocess_input(x)
    
    # Prediction: pass only the image to the model (no rarity input)
    prediction = card_id_model.predict(x)
    
    # Get the predicted card ID (index of highest probability)
    predicted_card_id_index = np.argmax(prediction[0])  # Assuming the highest output corresponds to the card ID
    predicted_card_id = card_ids[predicted_card_id_index]
    
    # Return the predicted set and card ID (no rarity needed)
    return predicted_set, predicted_card_id, img


# Example usage:

# Predict set and card ID
predicted_set, predicted_card_id, img = predict_card(imagePath)

if predicted_set is None:
    print("Prediction failed.")
else:
    # Display the image and predictions
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')  # Hide axis
    plt.title(f"Predicted Set: {predicted_set}\nPredicted Card ID: {predicted_card_id}")
    plt.show()
