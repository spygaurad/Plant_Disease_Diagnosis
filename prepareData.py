import os
import random
import csv
import re

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

    # Filter out files with purely numerical filenames from the test set
    filtered_test_files = []
    for file in test_files:
        if not re.match(r"^\d+\.jpg$", file):
            test_writer.writerow([os.path.join(folder_path, file), folder])
        else:
            filtered_test_files.append(file)

    # Distribute the excluded numerical files among train and valid sets
    num_excluded = len(filtered_test_files)
    num_train_excluded = int(num_excluded * 0.8)  # Ratio 4:1 for train and valid
    num_valid_excluded = num_excluded - num_train_excluded

    train_files.extend(filtered_test_files[:num_train_excluded])
    valid_files.extend(filtered_test_files[num_train_excluded:])

    # Shuffle the train and valid files separately
    random.shuffle(train_files, inplace=True)
    random.shuffle(valid_files, inplace=True)
    random.shuffle(test_files, inplace=True)

    # Write the file paths and class names to the train and valid CSV files
    for file in train_files:
        train_writer.writerow([os.path.join(folder_path, file), folder])

    for file in valid_files:
        valid_writer.writerow([os.path.join(folder_path, file), folder])

# Close the CSV files
train_file.close()
valid_file.close()
test_file.close()

print("CSV files created successfully.")



# import os
# import csv
# from shutil import copyfile

# csv_file = "Dataset/Plant_Village/test.csv"
# output_folder = "Test"

# # Create the output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# with open(csv_file, 'r') as file:
#     reader = csv.reader(file)
#     next(reader)  # Skip the header row
    
#     for row in reader:
#         file_path, class_name = row
#         class_folder = os.path.join(output_folder, class_name)
#         os.makedirs(class_folder, exist_ok=True)
        
#         image_name = os.path.basename(file_path)
#         output_path = os.path.join(class_folder, image_name)
        
#         copyfile(file_path, output_path)
#         print(f"Image saved: {output_path}")

# print("Processing complete.")
