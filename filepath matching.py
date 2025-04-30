import os
import csv
import math

def get_correct_image_filename(row):
    """Generate the correct image filename based on the row data."""
    name = row.get('name', '')  # Take name as is
    number = row.get('number', '')  # Take number as is
    rarity = row.get('rarity', '')  # Take rarity as is

    # If rarity is NaN, replace it with 'Unknown'
    if rarity == '' or rarity == 'nan' or rarity is None or str(rarity).lower() == 'nan' or isinstance(rarity, float) and math.isnan(rarity):
        rarity = 'Unknown'
    
    card_id = row.get('id', '')  # Take id as is
    # Construct the expected filename without modifying columns
    return f"{name}_{number}_{rarity}_{card_id}.jpg"

def check_and_correct_image_paths(csv_folder):
    # Go through each file in the folder and its subfolders
    for root, dirs, files in os.walk(csv_folder):
        for filename in files:
            if filename.endswith(".csv"):
                csv_path = os.path.join(root, filename)

                with open(csv_path, "r", newline="", encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    rows = list(reader)
                    columns = reader.fieldnames

                    # Show the columns in the current CSV
                    print(f"\nChecking {filename} for incorrect image paths...")
                    print(f"Columns: {columns}")

                    # Check image_path column for any discrepancies
                    discrepancies = []
                    for row in rows:
                        correct_filename = get_correct_image_filename(row)
                        image_path = row.get('image_path', '')

                        # Check if the current image path is valid and has the right filename
                        if image_path:
                            # Extract the filename from the path
                            path_parts = image_path.split(os.sep)
                            current_image_filename = path_parts[-1]

                            # Compare the filenames (ignoring the directory part)
                            if current_image_filename != correct_filename:
                                discrepancies.append({
                                    'row': row,
                                    'current_image_path': image_path,
                                    'correct_image_path': os.path.join(os.sep.join(path_parts[:-1]), correct_filename)  # Keep the directory part intact
                                })

                    if discrepancies:
                        print(f"\nFound {len(discrepancies)} discrepancies in {filename}:")

                        # Show discrepancies (first 5 rows)
                        for discrepancy in discrepancies[:5]:  # Show first 5 rows with discrepancies
                            print(f"Row: {discrepancy['row']}")
                            print(f"Current image path: {discrepancy['current_image_path']}")
                            print(f"Correct image path: {discrepancy['correct_image_path']}")
                            print()

                        # Ask for confirmation to apply changes
                        confirm = input(f"\nDo you want to correct the image paths in {filename}? (yes/no): ").strip().lower()
                        if confirm == "y":
                            # Correct the image paths in the rows
                            for discrepancy in discrepancies:
                                row = discrepancy['row']
                                row['image_path'] = discrepancy['correct_image_path']

                            # Write the corrected data back to the file
                            with open(csv_path, "w", newline="", encoding="utf-8") as file:
                                writer = csv.DictWriter(file, fieldnames=columns)
                                writer.writeheader()
                                writer.writerows(rows)

                            print(f"Corrected image paths in {filename}.")
                        else:
                            print(f"Kept the original image paths in {filename}.")
                    else:
                        print(f"No discrepancies found in {filename}.")

# Example usage
csv_folder_path = r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\combined_data_V3"
check_and_correct_image_paths(csv_folder_path)
