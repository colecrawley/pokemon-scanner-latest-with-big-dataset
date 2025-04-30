import os
import csv
import difflib
import re

def parse_card_number(num_str):
    """ Extract numeric and letter parts separately to ensure correct sorting. """
    match = re.match(r"(\d+)([a-zA-Z]*)", num_str)
    if match:
        num_part = int(match.group(1))  # Numeric part
        letter_part = match.group(2)    # Letter suffix (if any)
        return (num_part, letter_part)
    return (float("inf"), "")  # Default for non-numeric values (place at the end)

def check_and_preview_sorted_csv(csv_folder):
    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv") and not filename.endswith("_sorted.csv"):  # Ignore previously sorted copies
            csv_path = os.path.join(csv_folder, filename)

            with open(csv_path, "r", newline="", encoding="utf-8") as file:
                reader = csv.reader(file)
                header = next(reader)  # Read the header row

                # Ensure "number" column exists
                if "number" not in header:
                    print(f"\nSkipping {filename}: No 'number' column found.\n")
                    continue

                number_index = header.index("number")

                # Read and sort rows using improved parsing function
                rows = list(reader)
                sorted_rows = sorted(rows, key=lambda row: parse_card_number(row[number_index]))

                # Check if the file is already sorted by the number column
                if rows == sorted_rows:
                    print(f"\nSkipping {filename}: Already sorted by 'number'.\n")
                    continue

            print(f"\nPreview of changes for {filename}:")

            # Show the original rows and the sorted rows with differences
            print(f"\nOriginal rows (first 5 for preview):")
            for row in rows[:5]:  # Preview first 5 rows
                print(row)

            print(f"\nSorted rows (first 5 for preview):")
            for row in sorted_rows[:5]:  # Preview first 5 sorted rows
                print(row)

            # Convert rows to strings for diffing
            original_lines = ['\t'.join(row) + '\n' for row in rows]  # Convert each row to a tab-separated string
            sorted_lines = ['\t'.join(row) + '\n' for row in sorted_rows]  # Convert each sorted row

            # Show differences between the original and sorted rows
            diff = difflib.unified_diff(original_lines, sorted_lines, fromfile=filename, tofile=filename + "_sorted", lineterm="")

            print(f"\nDifferences for {filename}:")
            print("\n".join(diff))

            # Ask for confirmation to replace the original file
            confirm = input(f"\nDo you want to sort and replace the original {filename}? (y/n): ").strip().lower()
            if confirm == "y":
                # Write the sorted data back to the file
                with open(csv_path, "w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(header)  # Write the header first
                    writer.writerows(sorted_rows)  # Write sorted data

                print(f"\n{filename} has been replaced with the sorted version.\n")
            else:
                print(f"\nKept the original {filename}.\n")

# Example usage
csv_folder_path = r"C:\Users\Cole\Desktop\pokemon scanner latest with big dataset\combined_data_V3"
check_and_preview_sorted_csv(csv_folder_path)
