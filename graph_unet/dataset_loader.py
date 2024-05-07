import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = os.listdir(data_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.data_dir, img_name)
        # Open image and convert to RGB format
        image = Image.open(img_path).convert('RGB')
        # You can perform any additional preprocessing here if needed

        if self.transform:
            image = self.transform(image)

        return image
