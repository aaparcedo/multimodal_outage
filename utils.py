import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
import numpy as np


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
      return len(os.listdir(os.path.join(self.data_dir, "orange"))) - ( self.start_index * 2 )

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

# Graph WaveNet utilities:

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def load_adj(filename, adjtype):

    # Load the adjacency matrix from csv file

    if (filename.endswith('.csv')):
        adj_mx = pd.read_csv(filename, index_col=0)
        adj_mx = adj_mx.values
    else:
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(filename)

    if adjtype == "doubletransition":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"

    if (filename.endswith('.csv')):
        return None, None, adj
    else:
        return sensor_ids, sensor_id_to_ind, adj

# End of Graph WaveNet utilities.


def print_memory_usage():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9} GB")
