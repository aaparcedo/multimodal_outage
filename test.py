import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils import BlackMarbleDataset, mse_per_pixel, rmse_per_pixel, mae_per_pixel, mape_per_pixel, load_adj, load_checkpoint
from models.unet  import Modified_UNET
import pandas as pd
import numpy as np

dir_image = "/groups/mli/multimodal_outage/data/black_marble/hq/percent_normal/"

train_ia_id, test_m = {'h_ian': pd.Timestamp('2022-09-26'), 'h_idalia': pd.Timestamp('2023-08-30')}, {'h_michael': pd.Timestamp('2018-10-10')}
train_m_id, test_ia = {'h_michael': pd.Timestamp('2018-10-10'), 'h_idalia': pd.Timestamp('2023-08-30')}, {'h_ian': pd.Timestamp('2022-09-26')}
train_ia_m, test_id = {'h_ian': pd.Timestamp('2022-09-26'), 'h_michael': pd.Timestamp('2018-10-10')}, {'h_idalia': pd.Timestamp('2023-08-30')}

def test_model(st_gnn='gwnet', batch_size=1, horizon=7, size='S', job_id='test', ckpt_file_name='test', device='cuda', dataset=None):

  ckpt_folder_path = os.path.join(f'logs/{job_id}', 'ckpts')
  ckpt_path = os.path.join(ckpt_folder_path, ckpt_file_name)

  model = Modified_UNET(st_gnn=st_gnn).to(device=device)
  model = load_checkpoint(ckpt_path, model)

  transform = transforms.Compose([
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  # Load dataset
  if dataset is None:  
    dataset = BlackMarbleDataset(dir_image, size=size, start_index=horizon, transform=transform, case_study=test_m)

  print(f'Size of test dataset: {len(dataset)}')

  # Create data loaders
  loader_args = dict(batch_size=batch_size, num_workers=2)
  test_loader = DataLoader(dataset, shuffle=True, **loader_args)

  # Set up optimizer and custom loss function
  criterion = mse_per_pixel
  
  # Alternative Benchmarks
  rmse = rmse_per_pixel
  mae = mae_per_pixel
  mape = mape_per_pixel

  model.eval()
  test_loss = 0
  test_rmse = 0
  test_mae = 0
  test_mape = 0

  preds = [] # x
  real = [] # y
  
  with torch.no_grad():
    for item in test_loader:
      past_tensor, future_tensor = (tensor.to(device).permute(0, 2, 1, 3, 4, 5) for tensor in item)
      preds_tensor = model(past_tensor)
      loss = criterion(preds_tensor, future_tensor)
      test_rmse_loss = rmse(preds_tensor, future_tensor)
      test_mae_loss = mae(preds_tensor, future_tensor)
      test_mape_loss = mape(preds_tensor, future_tensor)
      
      test_loss += loss.item()
      test_rmse += test_rmse_loss.item()
      test_mae += test_mae_loss.item()
      test_mape += test_mape_loss.item()

      preds.append(preds_tensor)
      real.append(future_tensor)

  avg_test_loss = test_loss / len(test_loader)
  avg_test_rmse_loss = test_rmse / len(test_loader)
  avg_test_mae_loss = test_mae / len(test_loader)
  avg_test_mape_loss = test_mape / len(test_loader)
  
  test_metrics = {
    'loss': avg_test_loss,
    'rmse': avg_test_rmse_loss,
    'mae': avg_test_mae_loss,
    'mape': avg_test_mape_loss
  }
  
  if (visualize == True): 
    # TODO: need the correct directoy to save to.
    save_dir = ''
    batch, county, timestep, _, _ =  preds_tensor.shape
    # TODO: start_date -> need the start date at each batch to save accordingly.
    start_date = ''
    for b in batch: 
      for c in county: 
        for t in timestep: 
          numpy_array = preds_tensor[b, c, t].numpy()
          image = np.transpose(numpy_array, (1, 2, 0))
          plt.imshow(image)
          plt.axis('off')
          
          #TODO: current_date -> need to figure out how to add a date to my start date.
          current_date = '' 
          filename = os.path.join(save_dir, f"image_batch{b}_county{c}_timestep{t}.jpeg")
          filename = f"{current_date}.jpeg"
          plt.savefig(filename, format='jpeg', bbox_inches='tight', pad_inches=0)

  #print(f'Test Results; Loss (MSE): {avg_test_loss:.4f}, RMSE: {avg_test_rmse_loss:.4f}, MAPE: {avg_test_mape_loss:.4f}, MAE: {avg_test_mae_loss:.4f}')

  preds = torch.cat(preds, dim=0)
  reals = torch.cat(real, dim=0) 

  h_rmse_hist = []
  h_mae_hist = []
  h_mape_hist = []

  print(f'\nTest per horizon:')

  # Evaluate on each horizon
  for i in range(horizon):
    pred = preds[:, :, i, :, :, :]
    real = reals[:, :, i, :, :, :]

    h_rmse = rmse(pred, real)
    h_mae = mae(pred, real)
    h_mape = mape(pred, real) 

    print(f'Test on horizon {i+1}. RMSE={h_rmse}, MAE={h_mae}, MAPE={h_mape}')

    h_rmse_hist.append(h_rmse.cpu().item())
    h_mae_hist.append(h_mae.cpu().item())
    h_mape_hist.append(h_mape.cpu().item())

  print(f'Test results on average over {horizon} horizons:')
  print(f'RMSE={np.mean(h_rmse_hist)}, MAE={np.mean(h_mae_hist)}, MAPE={np.mean(h_mape_hist)}\n')

  return test_metrics  

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--st_gnn', dest='st_gnn', type=str, default='gwnet', help='Select st-gnn')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--horizon', dest='horizon', type=int, default=7, help='Timestep horizon')
    parser.add_argument('--size', dest='size', type=str, default='S', help='Dataset size/horizon')
    parser.add_argument('--job_id', dest='job_id', type=str, default='test', help='Slurm job ID')
    parser.add_argument('--device', dest='device', type=str, default='cuda', help='Select device, i.e., "cpu" or "cuda"')
    parser.add_argument('--checkpoint_path', dest='ckpt_path', type=str, help='Model checkpoint path to test')

    return parser.parse_args()


if __name__ == '__main__':
  args = get_args()
  print(f'\nargs: {args}\n')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  test_model(st_gnn=args.st_gnn, batch_size=args.batch_size, horizon=args.horizon, size=args.size, job_id=args.job_id, device=args.device, ckpt_file_name=args.ckpt_path)

