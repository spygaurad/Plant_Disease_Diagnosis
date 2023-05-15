import csv
import random
import os
import glob
from PIL import Image
from torchvision import transforms

# Function to apply random augmentations to an image
def apply_augmentations(image):
    # Define a list of transformations to apply
    transformations = [
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # Add more transformations as needed
    ]
    
    # Apply random transformations
    for transform in transformations:
        image = transform(image)
    
    return image

# Set the paths
dataset_path = 'Dataset/Plant_Village'
train_csv_path = 'Dataset/Plant_Village/train.csv'

# Create a list to store the new rows for train.csv
new_rows = []

# Open train.csv
with open(train_csv_path, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        file_path = row[0]
        class_name = row[1]
        
        # Open the image
        image_path = os.path.join(dataset_path, file_path)
        image = Image.open(image_path)
        
        # Apply augmentations
        augmented_image = apply_augmentations(image)
        
        # Generate a new random filename
        new_file_name = f'{random.randint(1, 100000)}.jpg'
        
        # Save the augmented image
        save_path = os.path.join(dataset_path, class_name, new_file_name)
        augmented_image.save(save_path)
        
        # Add a new row to train.csv
        new_row = [os.path.join(class_name, new_file_name), class_name]
        new_rows.append(new_row)

# Shuffle the new rows
random.shuffle(new_rows)

# Append the new rows to train.csv
with open(train_csv_path, 'a', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(new_rows)

print('Data augmentation and CSV update complete.')
