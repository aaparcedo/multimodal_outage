import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class BlackMarbleDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.county_paths = os.listdir(data_dir)
        self.transform = transform if transform is not None else transforms.ToTensor() 
   
        # Sorting each county's images by date
        self.sorted_image_paths = {
            county: sorted(os.listdir(os.path.join(data_dir, county)),
                           key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1]), int(x.split('_')[2].split('.')[0])))  # Improved date sorting
            for county in self.county_paths
        }
 
    def __len__(self):
      return len(self.sorted_image_paths['orange']) - 6
    
     

    def __getitem__(self, idx):
        if idx < 6:
            raise IndexError("Index too small. Must be at least 6 to retrieve a full week of data.")

        week_image_list = []  # To hold images from {idx-6} to {idx} for all 67 counties

        # Fetch images for the 7 days period
        for day in range(idx - 6, idx + 1):             
            day_image_list = []  # Hold images for one day from all counties
            for county in self.county_paths:
                county_path = os.path.join(self.data_dir, county)
                image_path = os.path.join(county_path, self.sorted_image_paths[county][day])

                if county == 'orange':
                    print(f'item idx: {day}')
                    print(image_path)

                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                day_image_list.append(image)
            
            week_image_list.append(torch.stack(day_image_list))  # Stack all county images for one day

        return torch.stack(week_image_list)
