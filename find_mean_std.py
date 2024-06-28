import torch
from torch.utils.data import DataLoader
from utils import BlackMarbleDataset
import pandas as pd

dir_image = "/groups/mli/multimodal_outage/data/black_marble/hq/original_gap_fill_rectangle"


train_ia_id, test_m = {'h_ian': pd.Timestamp('2022-09-26'), 'h_idalia': pd.Timestamp('2023-08-30')}, {'h_michael': pd.Timestamp('2018-10-10')}
train_m_id, test_ia = {'h_michael': pd.Timestamp('2018-10-10'), 'h_idalia': pd.Timestamp('2023-08-30')}, {'h_ian': pd.Timestamp('2022-09-26')}
train_ia_m, test_id = {'h_ian': pd.Timestamp('2022-09-26'), 'h_michael': pd.Timestamp('2018-10-10')}, {'h_idalia': pd.Timestamp('2023-08-30')}

entire_set = {'h_michael': pd.Timestamp('2018-10-10'), 'h_idalia': pd.Timestamp('2023-08-30'), 'h_ian': pd.Timestamp('2022-09-26')}

dataset = BlackMarbleDataset(dir_image, size='S', case_study=entire_set, horizon=1)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

# Initialize variables
mean = 0.0
sum_of_squares = 0.0
n_samples = 0
count = 0
for data in data_loader:
    past_tensor, future_tensor, _ = data

    # Use future_tensor for normalization
    data_tensor = future_tensor[0, 0, :, 0, :, :]  # Shape: [67, 128, 128]

    # Accumulate sum and sum of squares
    mean += data_tensor.sum().item()
    sum_of_squares += (data_tensor ** 2).sum().item()
    n_samples += data_tensor.numel()
    count+=1
    print(f'count: {count}')

# Compute mean
mean /= n_samples

# Compute variance and std
var = (sum_of_squares / n_samples) - (mean ** 2)
std = torch.sqrt(torch.tensor(var))

print(f'Mean: {mean}, Std: {std.item()}')

# Example of normalization
# Normalize the data tensor
normalized_tensor = (data_tensor - mean) / std

print(normalized_tensor)
print(f'Normalized Mean: {normalized_tensor.mean().item()}, Normalized Std: {normalized_tensor.std().item()}')
