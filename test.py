import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import torch.nn as nn
from utils import BlackMarbleDataset, mse_per_pixel, rmse_per_pixel, mae_per_pixel, mape_per_pixel, load_adj, load_checkpoint
from models.unet  import Modified_UNET


dir_image = "/groups/mli/multimodal_outage/data/black_marble/hq/percent_normal/"

def test_model(epochs=1, batch_size=1, horizon=7, size='S', job_id='test', ckpt_file_name='test', device='cuda', dataset=None):

  ckpt_folder_path = os.path.join(f'logs/{job_id}', 'ckpts')
  ckpt_path = os.path.join(ckpt_folder_path, ckpt_file_name)

  model = Modified_UNET().to(device=device)
  model = load_checkpoint(ckpt_path, model)

  transform = transforms.Compose([
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  # Load dataset
  if dataset is None:  
    dataset = BlackMarbleDataset(dir_image, size=size, start_index=horizon, transform=transform, evaluation=True)

  print(f'size of test dataset: {len(dataset)}')

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

  #print(f'Test Results; Loss (MSE): {avg_test_loss:.4f}, RMSE: {avg_test_rmse_loss:.4f}, MAPE: {avg_test_mape_loss:.4f}, MAE: {avg_test_mae_loss:.4f}')

  return test_metrics  

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--horizon', dest='horizon', type=int, default=7, help='Timestep horizon')
    parser.add_argument('--size', dest='size', type=str, default='S', help='Dataset size/horizon')
    parser.add_argument('--job_id', dest='job_id', type=str, default='test', help='Slurm job ID')
    parser.add_argument('--device', dest='device', type=str, default='cuda', help='Select device, i.e., "cpu" or "cuda"')
    parser.add_argument('--checkpoint_path', dest='chck_path', type=str, help='Model checkpoint path to test')

    return parser.parse_args()


if __name__ == '__main__':
  args = get_args()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  test_model(epochs=args.epochs, batch_size=args.batch_size, horizon=args.horizon, size=args.size, job_id=args.job_id, device=args.device)

