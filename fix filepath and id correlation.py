import pandas as pd
import os
import glob

# Base paths
csv_folder = r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\combined_data_V3"
new_base_path = r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\images_V3"

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

# Loop through all CSV files
for csv_path in csv_files:
    # Extract the set name from the CSV filename (removing .csv extension)
    set_name = os.path.splitext(os.path.basename(csv_path))[0]

    # Load CSV
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    required_columns = {'name', 'number', 'rarity', 'id'}
    if required_columns.issubset(df.columns):
        # Construct new image paths
        df['new_image_path'] = df.apply(lambda row: os.path.join(
            new_base_path, set_name,  
            f"{row['name']}_{row['number']}_{row['rarity']}_{row['id']}.jpg"
        ), axis=1)

        # Show all changes before proceeding
        print(f"\n=== Preview of changes for set: {set_name} ===")
        
        # Ensure full paths are visible
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_rows', None)  # Show all rows
        
        print(df[['image_path', 'new_image_path']])  # Show all rows

        # Ask for confirmation before saving
        proceed = input(f"\nSave changes for {set_name}? (y/n): ").strip().lower()
        if proceed == 'y':
            df.drop(columns=['image_path'], inplace=True)  # Remove old column
            df.rename(columns={'new_image_path': 'image_path'}, inplace=True)  # Rename new column
            df.to_csv(csv_path, index=False)
            print(f"✅ Changes saved for {set_name}. Moving to the next set.")
        else:
            print(f"❌ Skipping {set_name}. Moving to the next set.")

    else:
        print(f"⚠️ Skipping {csv_path} - Missing required columns. Found columns: {list(df.columns)}")
