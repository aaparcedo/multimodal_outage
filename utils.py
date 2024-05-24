import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import linalg
from normalization import normalize_and_transform_images


class BlackMarbleDataset(Dataset):
    def __init__(self, data_dir, size, case_study, start_index=7, transform=None):
        self.data_dir = data_dir
        self.size = size
        self.start_index = start_index
        self.county_paths = os.listdir(data_dir)
        self.transform = transform if transform is not None else transforms.ToTensor()

        # Sorting each county's images by date
        self.sorted_image_paths = {
            county: find_case_study_dates(
                size,
                sorted(os.listdir(os.path.join(data_dir, county)),
                       key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1]), int(x.split('_')[2].split('.')[0]))),
                case_study=case_study
            ) for county in self.county_paths
        }

    def __len__(self):
        return len(self.sorted_image_paths['orange'])

    def __iter__(self):
        return iter(range(self.start_index, len(self.data_dir) - self.start_index))

    def __getitem__(self, idx):

        past_image_list = []
        future_image_list = []

        # Fetch images for the start_index days period
        for day in range(self.start_index):
            past_days_image_list = []  # Hold images for one day from all counties
            future_days_image_list = []

            for county in self.county_paths:
                county_path = os.path.join(self.data_dir, county)
                past_image_path = os.path.join(
                    county_path, self.sorted_image_paths[county][day])
                future_image_path = os.path.join(
                    county_path, self.sorted_image_paths[county][day + self.start_index])

                past_image = Image.open(past_image_path).convert('RGB')
                future_image = Image.open(future_image_path).convert('RGB')

                if self.transform:
                    past_image = self.transform(past_image)
                    future_image = self.transform(future_image)

                past_days_image_list.append(past_image)
                future_days_image_list.append(future_image)

            # Stack all county images for one day
            past_image_list.append(torch.stack(past_days_image_list))
            future_image_list.append(torch.stack(future_days_image_list))

        past_image_tensor = torch.stack(past_image_list)
        future_image_tensor = torch.stack(future_image_list)

        # [batch_size, num_timesteps, num_nodes, num_channels, image_width, image_height]
        # [S, T, N, C, W, H], e.g., if batch_size = 1 and num_timesteps = 7, [1, 7, 67, 3, 128, 128]
        return (past_image_tensor, future_image_tensor)


def find_case_study_dates(size, image_paths, case_study):

    if size == 'S':
        horizon = 20  # or 90
    elif size == 'M':
        horizon = 60
    elif size == 'L':
        horizon = 90
    else:
        print('Invalid size. Please select a valid size, i.e., "S", "M", or "L"')

    timestamp_to_image = {pd.Timestamp(image_path.split('.')[0].replace(
        '_', '-')): image_path for image_path in image_paths}
    dates = [pd.Timestamp(image_path.split('.')[0].replace('_', '-'))
             for image_path in image_paths]

    case_study_indices = [dates.index(date) for date in case_study.values()]

    filtered_dates = set()
    main_directory = '/groups/mli/multimodal_outage/data/black_marble/hq/percent_normal'

    for case_study_index in case_study_indices:
        start_index = case_study_index - horizon
        end_index = case_study_index + horizon

        case_study_dates = dates[start_index:end_index]
        case_study_subset = dates[start_index:case_study_index]
        normalize_and_transform_images(
            main_directory=main_directory, dates=case_study_subset)

        filtered_dates.update(case_study_dates)

    filtered_image_paths = [timestamp_to_image[date]
                            for date in sorted(filtered_dates)]
    return filtered_image_paths


def mse_per_pixel(x, y):
    squared_diff = (x - y) ** 2
    total_mse = torch.mean(squared_diff)
    return total_mse


def rmse_per_pixel(x, y):
    squared_diff = (x - y) ** 2
    total_rmse = torch.sqrt(torch.mean(squared_diff))
    return total_rmse


def mae_per_pixel(x, y):
    error = x - y
    absolute_error = torch.abs(error)
    total_mae = torch.mean(absolute_error)
    return total_mae


def mape_per_pixel(x, y, epsilon=1e-8):
    return torch.mean(torch.abs((x - y) / (x + epsilon)))

# DCRNN utilities: 

def calculate_normalized_laplacian(adj):
  """
  # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
  # D = diag(A 1)
  :param adj:
  :return:
  """
  adj = sp.coo_matrix(adj)
  d = np.array(adj.sum(1))
  d_inv_sqrt = np.power(d, -0.5).flatten()
  d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
  d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
  normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
  return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
  adj_mx = sp.coo_matrix(adj_mx)
  d = np.array(adj_mx.sum(1))
  d_inv = np.power(d, -1).flatten()
  d_inv[np.isinf(d_inv)] = 0.
  d_mat_inv = sp.diags(d_inv)
  random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
  return random_walk_mx

def calculate_reverse_random_walk_matrix(adj_mx):
  return calculate_random_walk_matrix(np.transpose(adj_mx))

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
  if undirected:
      adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
  L = calculate_normalized_laplacian(adj_mx)
  if lambda_max is None:
      lambda_max, _ = linalg.eigsh(L, 1, which='LM')
      lambda_max = lambda_max[0]
  L = sp.csr_matrix(L)
  M, _ = L.shape
  I = sp.identity(M, format='csr', dtype=L.dtype)
  L = (2 / lambda_max * L) - I
  return L.astype(np.float32)

# Graph WaveNet utilities:

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
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


def plot_training_history(train_loss_hist, val_loss_hist, train_rmse_hist, val_rmse_hist,
                          train_mae_hist, val_mae_hist, train_mape_hist, val_mape_hist, save_path):
    epochs = range(1, len(train_loss_hist) + 1)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot training and validation Loss
    axs[0, 0].plot(epochs, train_loss_hist, label='Training Loss')
    axs[0, 0].plot(epochs, val_loss_hist, label='Validation Loss')
    axs[0, 0].set_title('Training and Validation Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # Plot training and validation RMSE
    axs[0, 1].plot(epochs, train_rmse_hist, label='Training RMSE')
    axs[0, 1].plot(epochs, val_rmse_hist, label='Validation RMSE')
    axs[0, 1].set_title('Training and Validation RMSE')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('RMSE')
    axs[0, 1].legend()

    # Plot training and validation MAE
    axs[1, 0].plot(epochs, train_mae_hist, label='Training MAE')
    axs[1, 0].plot(epochs, val_mae_hist, label='Validation MAE')
    axs[1, 0].set_title('Training and Validation MAE')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('MAE')
    axs[1, 0].legend()

    # Plot training and validation MAPE
    axs[1, 1].plot(epochs, train_mape_hist, label='Training MAPE')
    axs[1, 1].plot(epochs, val_mape_hist, label='Validation MAPE')
    axs[1, 1].set_title('Training and Validation MAPE')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('MAPE')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path)


def plot_error_metrics(runs_metrics, save_path):

    metrics = ['val_loss', 'val_rmse', 'val_mae', 'val_mape']
    # Extract event names from the provided dictionary
    events = list(runs_metrics.keys())

    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for col, event in enumerate(events):
        for row, metric in enumerate(metrics):
            for run in range(len(runs_metrics[event][metric])):
                axes[row, col].plot(runs_metrics[event]
                                    [metric][run], label=f'Run {run+1}')
            axes[row, col].set_title(f'{event} - {metric.upper()}')
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel(metric.upper())
            if row == 0:
                axes[row, col].legend()

    plt.savefig(save_path)


def save_checkpoint(model, optimizer, epoch, filename='checkpoint.pth.tar'):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {
          checkpoint_path}, starting from epoch {start_epoch}")
    return model


def print_memory_usage():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9} GB")
