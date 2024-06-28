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
import pickle

class BlackMarbleDataset(Dataset):
    def __init__(self, data_dir, size, case_study, horizon=7, transform=None):
        self.data_dir = data_dir
        self.size = size
        self.horizon = horizon
        self.county_names = sorted(os.listdir(data_dir))
        self.case_study = case_study

        # Sorting each county's images by date
        self.sorted_image_paths = {
          county: find_case_study_dates(
            size,
            sorted(os.listdir(os.path.join(data_dir, county)),
              key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1]), int(x.split('_')[2].split('.')[0]))),
            case_study=case_study  
          )  for county in self.county_names
        }
        
        self.mean = 3.201447427712248
        self.std = 10.389727592468262


        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def denormalize(self, tensor):
      mean = torch.tensor(self.mean).cuda()
      std = torch.tensor(self.std).cuda()

      return tensor * std + mean

    def open_pickle_as_tensor(self, image_path):
      with open(image_path, 'rb') as file:
        data = pickle.load(file)
      data_np = data["Gap_Filled_DNB_BRDF-Corrected_NTL"].values
      data_np[data_np == 6.5535e+03] = 0
      data_tensor = torch.Tensor(data_np).unsqueeze(0)
      return data_tensor

    def __len__(self):
        return len(self.sorted_image_paths[self.county_names[0]]) - self.horizon * 2

    def __getitem__(self, idx):
        past_image_list = []
        future_image_list = []
        time_embeds_list = []

        # Fetch images for the start_index days period
        for day in range(self.horizon):
            past_days_county_image_list = []  # Hold images for one day from all counties
            future_days_county_image_list = []

            for county in self.county_names:
                county_path = os.path.join(self.data_dir, county)
                past_image_path = os.path.join(
                    county_path, self.sorted_image_paths[county][day + idx])
                future_image_path = os.path.join(
                    county_path, self.sorted_image_paths[county][day + idx +  self.horizon])

                past_image = self.open_pickle_as_tensor(past_image_path)
                future_image = self.open_pickle_as_tensor(future_image_path)

                if self.transform:
                    past_image = self.transform(past_image)
                    future_image = self.transform(future_image)

                past_days_county_image_list.append(past_image)
                future_days_county_image_list.append(future_image)

            time_embed = generate_Date2Vec(past_image_path)
            time_embeds_list.append(time_embed)

            # Stack all county images for one day
            past_image_list.append(torch.stack(past_days_county_image_list))
            future_image_list.append(torch.stack(future_days_county_image_list))

        past_image_tensor = torch.stack(past_image_list)
        future_image_tensor = torch.stack(future_image_list)
        time_embeds = torch.stack(time_embeds_list).view(1, self.horizon, 64).repeat(67, 1, 1) # [67, horizon, 64] 

        return (past_image_tensor, future_image_tensor, time_embeds) 


from Model import Date2VecConvert
d2v = Date2VecConvert(model_path="./d2v_model/d2v_98291_17.169918439404636.pth")

def generate_Date2Vec(filepath):
  """
  Generates time embeddings given a list of dates.
  Paper: https://arxiv.org/abs/1907.05321
  Code: https://github.com/ojus1/Date2Vec
  """

  year, month, day = filepath.split('/')[-1].split('.')[0].split('_')
  
  x = torch.Tensor([[00, 00, 00, int(year), int(month), int(day)]]).float()

  time_embeds = d2v(x)
  return time_embeds


def find_case_study_dates(size, image_paths, case_study):
    
    if size == 'test':
      horizon = 15
    elif size == 'S':
      horizon = 30 # or 90 
    elif size == 'M':
      horizon = 60
    elif size == 'L':
      horizon = 90
    elif size == 'XL':
      horizon = 120
    elif size == 'XXL':
      horizon = 150
    elif size == 'XXXL':
      horizon = 180
    else:
      print('Invalid size. Please select a valid size, i.e., "S", "M", or "L"')

    timestamp_to_image = {pd.Timestamp(image_path.split('.')[0].replace('_', '-')): image_path for image_path in image_paths}
    dates = [pd.Timestamp(image_path.split('.')[0].replace('_', '-')) for image_path in image_paths]
    case_study_indices = [dates.index(date) for date in case_study.values()]
    filtered_dates = set()

    for case_study_index in case_study_indices:
        start_index = case_study_index - horizon
        end_index = case_study_index + horizon

        case_study_dates = dates[start_index:end_index]

        filtered_dates.update(case_study_dates)
    filtered_image_paths = [timestamp_to_image[date] for date in sorted(filtered_dates)]
    return filtered_image_paths


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
    events = list(runs_metrics.keys())
    
    # Calculate vmin and vmax for each metric
    metric_min_max = {}
    for metric in metrics:
        all_values = []
        for event in events:
            for run_data in runs_metrics[event][metric]:
                all_values.extend(run_data)
        metric_min_max[metric] = (min(all_values), max(all_values))
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    for col, event in enumerate(events):
        for row, metric in enumerate(metrics):
            for run in range(len(runs_metrics[event][metric])):
                axes[row, col].plot(runs_metrics[event][metric][run], label=f'Run {run+1}')
            axes[row, col].set_title(f'{event} - {metric.upper()}')
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel(metric.upper())
            axes[row, col].set_ylim(metric_min_max[metric])  # Set the same vmin and vmax for the metric
            if row == 0:
                axes[row, col].legend()

    plt.savefig(save_path)

def ntl_tensor_to_np(ntl, dataset=None, denorm=True):
  if denorm:
    ntl = dataset.denormalize(ntl).cpu()
  
  ntl_np = np.array(ntl)	
  ntl_np = np.transpose(ntl_np, (0, 2, 1))
  ntl_np = np.rot90(ntl_np, k=1, axes=(1, 2))
  ntl_np = ntl_np[0, :, :]
  return ntl_np

def visualize_results_raster(preds, save_dir, save_folder, dataset_dir, dataset):
  """
  Save qualitative results from VST-GNN predictions.
  """
  
  eid_path = os.path.dirname(os.path.dirname(save_dir))

  county_names = sorted(os.listdir(dataset_dir))
  preds_save_dir = os.path.join(eid_path, save_folder)
  os.makedirs(preds_save_dir, exist_ok=True) 
 
  case_study_county_idx = [2, 34, 36]
 
  for pred_idx in range(preds.shape[0]):
    for pred_horizon in range(preds.shape[2]):

      pred_horizon_folder_path = os.path.join(preds_save_dir, str(pred_horizon + 1))
      os.makedirs(pred_horizon_folder_path, exist_ok=True)

      for county_idx in case_study_county_idx:
        county_horizon_folder_path = os.path.join(pred_horizon_folder_path, county_names[county_idx])
        os.makedirs(county_horizon_folder_path, exist_ok=True)

        pred_input_filename = dataset.sorted_image_paths[county_names[county_idx]][pred_idx + pred_horizon + dataset.horizon].split('.')[0]
        pred_save_path = os.path.join(county_horizon_folder_path, pred_input_filename)

        pred = preds[pred_idx, county_idx, pred_horizon]
        pred_np = ntl_tensor_to_np(pred, dataset)

        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        c = ax.pcolormesh(pred_np, shading='auto', cmap="cividis")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(pred_save_path, bbox_inches='tight')
        plt.close()


def get_percent_of_normal_ntl(ntl, filename, county_name):
  """
  ntl (np.array): radiance data
  filename (str): filename to find correct "normal", i.e., look up the correct month
  county_name (str):
  """
  
  month_composites = load_month_composites(county_name)
  m_ntl = calculate_average_month_ntl(filename, month_composites)
  pon_ntl = 100 * ( (ntl + 1) / (m_ntl + 1) )
  return pon_ntl
  

def calculate_average_month_ntl(filename, month_composites):
  """
  Calculates the average monthly composite of the last three months from a given date.

  Parameters:
  - filename (str):
  - month_composites (xarray.Dataset): object containing necessary monthly composites

  Returns:
  avg_month_ntl (np.ndarray): represents the last 3 month average ntl
  """

  date = pd.Timestamp(filename.split('.')[0].replace('_', '-'))
  transform = transforms.Resize((128, 128))

  if date.year == 2018: 
    month_list = ['2018-06-01', '2018-07-01', '2018-08-01']
  elif date.year == 2022:
    month_list = ['2022-06-01', '2022-07-01', '2022-08-01']
  elif date.year == 2023:
    month_list = ['2023-04-01', '2023-05-01', '2023-06-01']
  else:
    print('Invalid date')

  monthly_ntl = []
  for month in month_list:
    month_ntl = month_composites["NearNadir_Composite_Snow_Free"].sel(time=month).values
    month_ntl[month_ntl == 6.5535e+03] = 0
     
    # convert to tensor to use transforms.Resize -> convert back to np
    month_ntl_tensor = transform(torch.Tensor(month_ntl).unsqueeze(0))
    month_ntl = ntl_tensor_to_np(month_ntl_tensor, denorm=False)
    monthly_ntl.append(month_ntl)

  avg_month_ntl = np.mean(monthly_ntl, axis=0)
 
  return avg_month_ntl


def load_month_composites(county_name):
  """
  Loads all the available monthly composites into memory.

  Parameters:
  - county_name (str): name of county, e.g., 'orange'

  Returns:
  - month_composites (xarray.Dataset): dataset of monthly composites
  """

  base_dir = "/groups/mli/multimodal_outage/data/black_marble/hq/monthly_bbox"
  county_dir = os.path.join(base_dir, county_name)
  file_path = os.path.join(county_dir, f"{county_name}.pickle")
  with open(file_path, 'rb') as file:
    month_composites = pickle.load(file)

  return month_composites


def visualize_risk_map(ntls, save_dir, save_folder, dataset):
  eid_path = os.path.dirname(os.path.dirname(save_dir))

  county_names = sorted(os.listdir(dataset.data_dir))
  save_dir = os.path.join(eid_path, save_folder)
  os.makedirs(save_dir, exist_ok=True)

  case_study_county_idx = [2, 34, 36]

  for idx in range(ntls.shape[0]):
    for horizon in range(ntls.shape[2]):

      horizon_folder_path = os.path.join(save_dir, str(horizon + 1))
      os.makedirs(horizon_folder_path, exist_ok=True)

      for county_idx in case_study_county_idx:
        county_horizon_folder_path = os.path.join(horizon_folder_path, county_names[county_idx])
        os.makedirs(county_horizon_folder_path, exist_ok=True)

        filename = dataset.sorted_image_paths[county_names[county_idx]][idx + horizon + dataset.horizon].split('.')[0]
        save_path = os.path.join(county_horizon_folder_path, filename)

        ntl = ntls[idx, county_idx, horizon]
        ntl_np = ntl_tensor_to_np(ntl, dataset, denorm=True)
        pon_ntl = get_percent_of_normal_ntl(ntl_np, filename, county_names[county_idx])

        # plot using red-yellow-green color map
        fig, ax = plt.subplots(figsize=(10, 10))
        c = ax.pcolormesh(pon_ntl, shading='auto', cmap="RdYlGn", vmin=0, vmax=100)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis("off")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


def visualize_results_color(preds, reals, save_dir, dataset_dir, dataset):
  """
  Save image results from modified unet predictions.

  Parameters:
  - preds (torch.Tensor): output predictions from modified unet model
  - save_dir (str): directory of to save images
  - dataset_dir (str): directory of dataset images
  - dataset (BlackMarbleDataset): dataset object  

  Returns:
  - N/A
  """

  county_names = sorted(os.listdir(dataset_dir))
  preds_save_dir = os.path.join(save_dir, 'preds')
  reals_save_dir = os.path.join(save_dir, 'reals')
  os.makedirs(preds_save_dir, exist_ok=True)
  os.makedirs(reals_save_dir, exist_ok=True)

  for pred_idx in range(preds.shape[0]):
    for horizon in range(preds.shape[2]):

      horizon_folder_path = os.path.join(preds_save_dir, str(horizon + 1)) # /logs/job_id/test_preds/horizon/
      reals_horizon_folder_path = os.path.join(reals_save_dir, str(horizon + 1))

      os.makedirs(horizon_folder_path, exist_ok=True)
      os.makedirs(reals_horizon_folder_path, exist_ok=True)

      for county_idx in range(preds.shape[1]):
        county_horizon_folder_path = os.path.join(horizon_folder_path, county_names[county_idx]) # /logs/job_id/test_preds/horizon/county/
        os.makedirs(county_horizon_folder_path, exist_ok=True)

        image_name = dataset.sorted_image_paths[county_names[county_idx]][pred_idx + horizon + dataset.start_index]
        image_save_path = os.path.join(county_horizon_folder_path, image_name)

        image_tensor = preds[pred_idx, county_idx, horizon]
        image_np = dataset.denormalize(image_tensor[0]).cpu().numpy()
        image_np = np.clip(image_np, 0, 1)
        image_np_uint8 = (image_np * 255).astype(np.uint8)
        image = Image.fromarray(image_np_uint8, mode='L')

        image.save(image_save_path)
        
        if reals is not None:
          reals_county_horizon_folder_path = os.path.join(reals_horizon_folder_path, county_names[county_idx])
          os.makedirs(reals_county_horizon_folder_path, exist_ok=True)
          real_image_save_path = os.path.join(reals_county_horizon_folder_path, image_name)
          image_tensor = reals[pred_idx, county_idx, horizon]
          image_np = dataset.denormalize(image_tensor[0]).cpu().numpy()
          image_np = np.clip(image_np, 0, 1)
          image_np_uint8 = (image_np * 255).astype(np.uint8)
          image = Image.fromarray(image_np_uint8, mode='L')
          image.save(real_image_save_path)


def print_memory_usage():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9} GB")
