import torch
from torch.utils.data import DataLoader
from utils import BlackMarbleDataset
import pandas as pd

ntl_dir = "/groups/mli/multimodal_outage/data/black_marble/hq/ntl"
pon_dir = "/groups/mli/multimodal_outage/data/black_marble/hq/percent_normal/"
pickle_dir = "/groups/mli/multimodal_outage/data/black_marble/hq/original"
ntl_gray_dir = "/groups/mli/multimodal_outage/data/black_marble/hq/ntl_gray"
dir_image = ntl_gray_dir


train_ia_id, test_m = {'h_ian': pd.Timestamp('2022-09-26'), 'h_idalia': pd.Timestamp('2023-08-30')}, {'h_michael': pd.Timestamp('2018-10-10')}
train_m_id, test_ia = {'h_michael': pd.Timestamp('2018-10-10'), 'h_idalia': pd.Timestamp('2023-08-30')}, {'h_ian': pd.Timestamp('2022-09-26')}
train_ia_m, test_id = {'h_ian': pd.Timestamp('2022-09-26'), 'h_michael': pd.Timestamp('2018-10-10')}, {'h_idalia': pd.Timestamp('2023-08-30')}

entire_set = {'h_michael': pd.Timestamp('2018-10-10'), 'h_idalia': pd.Timestamp('2023-08-30'), 'h_ian': pd.Timestamp('2022-09-26')}


dataset = BlackMarbleDataset(dir_image, size='S', case_study=entire_set)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

mean = 0.0
std = 0.0
n_samples = 0

# future_tensor shape: torch.Size([1, 7, 67, 3, 128, 128])

for data in data_loader:



    past_tensor, future_tensor, _ = data

    # [1, 7, 67, 1, 128, 128] 
    data_tensor = future_tensor  # Use future_tensor for normalization

    # [67, 1, 128, 128]
    data_tensor = data_tensor[0, 0, :, 0, :, :]

    mean += data_tensor.mean(dim=(0, 1, 2))
    std += data_tensor.std(dim=(0, 1, 2))

    n_samples += 1

    print(n_samples)

mean /= n_samples
std /= n_samples

print(f"Mean: {mean}")
print(f"Std: {std}")
print(f'Dataset size: {dataset.size}')
print(f'Train set: {dataset.case_study}')
