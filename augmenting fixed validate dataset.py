from PIL import ImageEnhance, Image
import os
import numpy as np
import random
import matplotlib.pyplot as plt

def adjust_brightness(image_path, factor):
    img = Image.open(image_path).convert('RGB')
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

# Paths
validate_path = r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\dataset_for_model\validate"
output_path = r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\dataset_for_model\validate_augmented"

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Process images
for class_folder in os.listdir(validate_path):
    class_path = os.path.join(validate_path, class_folder)
    output_class_path = os.path.join(output_path, class_folder)
    os.makedirs(output_class_path, exist_ok=True)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        # Skip non-image files
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Apply brightness change (stronger range for testing)
        brightness_factor = np.random.uniform(0.7, 1.3)  # Increased range
        new_img = adjust_brightness(img_path, brightness_factor)

        # Save the modified image
        new_img.save(os.path.join(output_class_path, img_name))

        # Debugging: Print brightness factor applied
        print(f"Applied brightness factor {brightness_factor:.2f} to {img_name}")

print("✅ Validation images augmented successfully!")

# ------------------------------
# DISPLAY BEFORE & AFTER IMAGES
# ------------------------------

# Select a random class folder
random_class = random.choice(os.listdir(validate_path))

# Get images from the original and augmented folders
original_class_path = os.path.join(validate_path, random_class)
augmented_class_path = os.path.join(output_path, random_class)

original_images = [f for f in os.listdir(original_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
augmented_images = [f for f in os.listdir(augmented_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Select 5 random images
num_samples = min(5, len(original_images))  
random_images = random.sample(original_images, num_samples)

# Plot images
plt.figure(figsize=(10, 5))
for i, img_name in enumerate(random_images):
    original_img = Image.open(os.path.join(original_class_path, img_name))
    augmented_img = Image.open(os.path.join(augmented_class_path, img_name))

    # Display original
    plt.subplot(2, num_samples, i + 1)
    plt.imshow(original_img)
    plt.axis('off')
    plt.title("Original")

    # Display augmented
    plt.subplot(2, num_samples, i + num_samples + 1)
    plt.imshow(augmented_img)
    plt.axis('off')
    plt.title("Augmented")

plt.suptitle("🔆 Brightness Augmentation Before & After")
plt.tight_layout()
plt.show()
