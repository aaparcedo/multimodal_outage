import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


class BlackMarbleDataset(Dataset):
    def __init__(self, data_dir, start_index=7, transform=None):
        self.data_dir = data_dir
        self.start_index = start_index
        self.county_paths = os.listdir(data_dir)
        self.transform = transform if transform is not None else transforms.ToTensor() 
   
        # Sorting each county's images by date
        self.sorted_image_paths = {
            county: sorted(os.listdir(os.path.join(data_dir, county)),
                           key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1]), int(x.split('_')[2].split('.')[0])))  # Improved date sorting
            for county in self.county_paths
        }
 
    def __len__(self):
      return len(self.data_dir) - ( self.start_index * 2 )
    
    def __iter__(self):
      return iter(range(self.start_index, len(self.data_dir)))

    def __getitem__(self, idx):

      past_image_list = [] 
      future_image_list = []

      # Fetch images for the start_index days period
      for day in range(self.start_index):             
        past_days_image_list = []  # Hold images for one day from all counties
        future_days_image_list = []

        for county in self.county_paths:
          county_path = os.path.join(self.data_dir, county)
          past_image_path = os.path.join(county_path, self.sorted_image_paths[county][day])
          future_image_path = os.path.join(county_path, self.sorted_image_paths[county][day + self.start_index])
          
          past_image = Image.open(past_image_path).convert('RGB')
          future_image = Image.open(future_image_path).convert('RGB')
          
          if self.transform:
            past_image = self.transform(past_image)
            future_image = self.transform(future_image)

          past_days_image_list.append(past_image)
          future_days_image_list.append(future_image)

        past_image_list.append(torch.stack(past_days_image_list))  # Stack all county images for one day
        future_image_list.append(torch.stack(future_days_image_list))
      
      past_image_tensor = torch.stack(past_image_list)
      future_image_tensor = torch.stack(future_image_list)
      
      # [batch_size, num_timesteps, num_nodes, num_channels, image_width, image_height]
      # [S, T, N, C, W, H], e.g., if batch_size = 1 and num_timesteps = 7, [1, 7, 67, 3, 128, 128]
      return (past_image_tensor, future_image_tensor)


#dataset = BlackMarbleDataset(data_dir, start_index=7)

#dataloader = DataLoader(dataset, batch_size=1)
