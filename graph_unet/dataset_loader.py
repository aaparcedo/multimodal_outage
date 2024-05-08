import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class BlackMarbleDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.county_paths = os.listdir(data_dir)
             
    def __len__(self):
        return len(os.listdir(os.path.join(self.data_dir, 'orange')))

    def __getitem__(self, idx):
      day_list = [] # hold image for {idx} for all 67 counties     

      for county in self.county_paths:
        county_path = os.path.join(self.data_dir, county)
        image_paths = os.listdir(county_path)
        image_path = os.path.join(county_path, image_paths[idx])  
        image = Image.open(image_path).convert('RGB')
        day_list.append(image)

      return day_list
