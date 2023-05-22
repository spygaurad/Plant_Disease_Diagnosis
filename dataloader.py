import os
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms, ToTensor, RandomRotation, RandomAffine, Resize, Normalize
from PIL import Image



class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.transform = transform
        self.data = []
        self.class_to_idx = {
            "Tomato Bacterial Spot": 0, 
            "Tomato Early Blight": 1, 
            "Tomato Healthy": 2, 
            "Tomato Late Blight": 3, 
            "Tomato Leaf Mold": 4, 
            "Tomato Septoria Leaf Spot": 5, 
            "Tomato Spider Mites": 6, 
            "Tomato Target Spot": 7, 
            "Tomato Mosaic Virus": 8,
            "Tomato Yellow Leaf Curl Virus": 9,
        }
        with open(csv_file, 'r') as f:
            for row in f:
                file_path, label = row.split(',')
                label = label.strip()
                label_idx = self.class_to_idx[label]
                self.data.append((file_path, label_idx))

    def __getitem__(self, index):
        file_path, label = self.data[index]
        try:
            image = Image.open(file_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(file_path)

    def __len__(self):
        return len(self.data)



def get_dataloader(root_dir, batch_size):

    train_transform = transforms.Compose([
        # RandomRotation(degrees=45), 
        # RandomAffine(degrees=0, shear=10),  # Random skewness and shear up to 10 degrees
        Resize((224, 224)),  # Resize to 224x224
        ToTensor(),  # Convert to tensor
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        Resize((224, 224)),
        ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CustomDataset(os.path.join(root_dir, "train.csv"), transform=train_transform)
    val_dataset = CustomDataset(os.path.join(root_dir, "valid.csv"), transform=val_transform)  # Resize validation images to 224x224
    test_dataset = CustomDataset(os.path.join(root_dir, "test.csv"), transform=val_transform)  # Resize validation images to 224x224

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # for element in train_loader:
    #     print(element)
    return train_loader, val_loader, test_loader



# train_data, val_data, test_data = get_dataloader("Dataset/Plant_Village/", batch_size=4)
