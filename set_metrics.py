import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
MODEL_PATH = r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\dataset_for_model\setModel.h5"
VALIDATE_FOLDER = r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\dataset_for_model\validate"
SAVE_FOLDER = r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\dataset_for_model\metrics_for_set_model\more_metrics_for_diss"

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded from: {MODEL_PATH}")

# Create the folder to save metrics if it doesn't exist
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Prepare validation data generator
valid_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

valid_generator = valid_datagen.flow_from_directory(
    VALIDATE_FOLDER,
    target_size=(224, 224),
    batch_size=1,
    shuffle=False,  # Don't shuffle
    class_mode='categorical'
)

# Get true labels
y_true = valid_generator.classes
class_indices = valid_generator.class_indices  # {class_name: index}
index_to_class = {v: k for k, v in class_indices.items()}  # {index: class_name}

# Predict
y_pred_probs = model.predict(valid_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# Metrics
acc = accuracy_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=[index_to_class[i] for i in range(len(index_to_class))], digits=4)

# Calculate per-class accuracy
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

# Preview metrics
print("\nSet Model Metrics:")
print(f"Overall Accuracy: {acc * 100:.2f}%")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print("\nPer-Class Accuracy:")
for i, class_name in index_to_class.items():
    print(f"{class_name}: {per_class_accuracy[i] * 100:.2f}%")
print("\nPrecision, Recall, F1-Score per class:")
print(report)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=[index_to_class[i] for i in range(len(index_to_class))],
            yticklabels=[index_to_class[i] for i in range(len(index_to_class))])
plt.title('Confusion Matrix for Set Model')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()

# Save confusion matrix as image
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=[index_to_class[i] for i in range(len(index_to_class))],
            yticklabels=[index_to_class[i] for i in range(len(index_to_class))])
plt.title('Confusion Matrix for Set Model')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_FOLDER, 'confusion_matrix_set_model.png'))
plt.close()

# Save all metrics to a text file
with open(os.path.join(SAVE_FOLDER, 'metrics_set_model.txt'), 'w') as f:
    f.write(f"Overall Accuracy: {acc * 100:.2f}%\n")
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n\n")
    f.write("Per-Class Accuracy:\n")
    for i, class_name in index_to_class.items():
        f.write(f"{class_name}: {per_class_accuracy[i] * 100:.2f}%\n")
    f.write("\nPrecision, Recall, F1-Score per class:\n")
    f.write(report)

print(f"\nAll metrics and confusion matrix saved to: {SAVE_FOLDER}")
