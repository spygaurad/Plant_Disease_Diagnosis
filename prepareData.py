import os
import random
import csv

dataset_folder = "Dataset/Plant_Village"
output_folder = "Dataset"
train_ratio = 0.7
valid_ratio = 0.2
test_ratio = 0.1

# Get the list of folders in the dataset folder
folders = [folder for folder in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, folder))]

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the CSV files for writing
train_file = open(os.path.join(output_folder, "train.csv"), "w", newline="")
valid_file = open(os.path.join(output_folder, "valid.csv"), "w", newline="")
test_file = open(os.path.join(output_folder, "test.csv"), "w", newline="")

train_writer = csv.writer(train_file)
valid_writer = csv.writer(valid_file)
test_writer = csv.writer(test_file)

# Write the headers to the CSV files
train_writer.writerow(["File Path", "Class"])
valid_writer.writerow(["File Path", "Class"])
test_writer.writerow(["File Path", "Class"])

# Iterate over the folders
for folder in folders:
    folder_path = os.path.join(dataset_folder, folder)
    files = os.listdir(folder_path)

    # Shuffle the files
    random.shuffle(files)

    num_files = len(files)
    num_train = int(num_files * train_ratio)
    num_valid = int(num_files * valid_ratio)
    num_test = num_files - num_train - num_valid

    # Split the files into train, valid, and test sets
    train_files = files[:num_train]
    valid_files = files[num_train:num_train + num_valid]
    test_files = files[num_train + num_valid:]

    # Write the file paths and class names to the respective CSV files
    for file in train_files:
        train_writer.writerow([os.path.join(folder_path, file), folder])

    for file in valid_files:
        valid_writer.writerow([os.path.join(folder_path, file), folder])

    for file in test_files:
        test_writer.writerow([os.path.join(folder_path, file), folder])

# Close the CSV files
train_file.close()
valid_file.close()
test_file.close()

print("CSV files created successfully.")
