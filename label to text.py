import os
import pandas as pd

# Get the set labels from categories (assuming categories are sorted)
categories = os.listdir(r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\dataset_for_model\train")
categories.sort()

# Save the set labels to a text file
with open(r"C:\Users\Cole\Desktop\set_labels.txt", "w") as f:
    for label in categories:
        f.write(f"{label}\n")
print("Set labels saved to set_labels.txt")

# Function to get card IDs from the CSV file of the predicted set
def get_card_ids_from_csv(set_name):
    csv_path = os.path.join(r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\151_data", f"{set_name}_cards.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found for set: {set_name}")
        return None
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Get the card IDs
    card_ids = df['id'].values  # Assuming 'id' column exists in your CSV
    
    return card_ids

# Save card ID labels for each set to individual text files
for set_name in categories[:5]:  # First 5 sets
    card_ids = get_card_ids_from_csv(set_name)
    if card_ids is not None:
        file_path = f"C:\\Users\\Cole\\Desktop\\card_id_labels_{set_name}.txt"
        with open(file_path, "w") as f:
            for card_id in card_ids:
                f.write(f"{card_id}\n")
        print(f"Card ID labels for {set_name} saved to {file_path}")
    else:
        print(f"Card ID labels for {set_name} could not be saved.")
