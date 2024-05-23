import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from tqdm import tqdm
from torchvision import transforms
import os
import torch.nn as nn
from utils import BlackMarbleDataset, mse_per_pixel, rmse_per_pixel, mae_per_pixel, mape_per_pixel, load_adj, print_memory_usage, plot_training_history, save_checkpoint
from models.unet  import Modified_UNET


dir_image = "/groups/mli/multimodal_outage/data/black_marble/hq/percent_normal/"


def train_model(epochs=1, batch_size=1, horizon=7, size='S', job_id='test', ckpt_file_name='test', device='cuda', dataset=None):

  randomadj = True
  adjdata = "/home/aaparcedo/multimodal_outage/data/graph/adj_mx_fl_k1.csv"
  adjtype = "doubletransition"

  sensor_ids, sensor_id_to_ind, adj_mx = load_adj(adjdata,adjtype)  
  supports = [torch.tensor(i).to(device) for i in adj_mx]

  if randomadj:
    adjinit = None
  else:
    adjinit = supports[0]

  model = Modified_UNET(supports).to(device=device)

  transform = transforms.Compose([
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  print(f'device: {device}')

  # Load dataset
  
  if dataset is None:
    dataset = BlackMarbleDataset(dir_image, size=size, transform=transform, start_index=horizon)

  print(f'size of dataset: {len(dataset)}')

  n_val = int(len(dataset) * 0.3)
  n_train = len(dataset) - n_val
  train_set, val_set= random_split(dataset, [n_train, n_val])

  # Create data loaders
  loader_args = dict(batch_size=batch_size, num_workers=2)
  train_loader = DataLoader(train_set, shuffle=True, **loader_args)
  val_loader = DataLoader(val_set, shuffle=True, **loader_args)
  #test_loader = DataLoader(test_set, shuffle=True, **loader_args)

  # Set up optimizer and custom loss function
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  criterion = mse_per_pixel
  
  # Alternative Benchmarks
  rmse = rmse_per_pixel
  mae = mae_per_pixel
  mape = mape_per_pixel
 
  train_val_metrics = {
    'train_loss': [],
    'val_loss': [],
    'train_rmse': [],
    'val_rmse': [],
    'train_mae': [],
    'val_mae': [],
    'train_mape': [],
    'val_mape': []
  }

  best_val_loss = float('inf') 

  # Begin training
  for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    train_rmse = 0
    train_mae = 0
    train_mape = 0
    
    with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='day') as pbar:
   
      # item is a tensor of shape [67, 3, 128, 128]
      for item in train_loader:
        past_tensor, future_tensor = (tensor.to(device).permute(0, 2, 1, 3, 4, 5) for tensor in item)
        preds_tensor = model(past_tensor)

        # pixel-wise MSE 
        loss = criterion(preds_tensor, future_tensor)
        
        # pixel-wise RMSE, MAE & MAPE
        with torch.no_grad():
          rmse_loss = rmse(preds_tensor, future_tensor)
          mae_loss = mae(preds_tensor, future_tensor)
          mape_loss = mape(preds_tensor, future_tensor)        
        train_rmse += rmse_loss.item()
        train_mae += mae_loss.item()
        train_mape += mape_loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pbar.update(past_tensor.shape[0])    
        epoch_loss += loss.item()
        pbar.set_postfix({'loss (batch)': loss.item()})

    model.eval()
    val_loss = 0
    val_rmse = 0
    val_mae = 0
    val_mape = 0

    with torch.no_grad():
      for item in val_loader:
        past_tensor, future_tensor = (tensor.to(device).permute(0, 2, 1, 3, 4, 5) for tensor in item)
        preds_tensor = model(past_tensor)

        loss = criterion(preds_tensor, future_tensor)
        val_rmse_loss = rmse(preds_tensor, future_tensor)
        val_mae_loss = mae(preds_tensor, future_tensor)
        val_mape_loss = mape(preds_tensor, future_tensor)
       
        val_loss += loss.item() 
        val_rmse += val_rmse_loss.item()
        val_mae += val_mae_loss.item()
        val_mape += val_mape_loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    avg_train_rmse_loss = train_rmse / len(train_loader)
    avg_train_mae_loss = train_mae / len(train_loader)
    avg_train_mape_loss = train_mape / len(train_loader)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_rmse_loss = val_rmse / len(val_loader)
    avg_val_mae_loss = val_mae / len(val_loader)
    avg_val_mape_loss = val_mape / len(val_loader)

    train_val_metrics['train_loss'].append(avg_train_loss)
    train_val_metrics['val_loss'].append(avg_val_loss)
    train_val_metrics['train_rmse'].append(avg_train_rmse_loss)
    train_val_metrics['val_rmse'].append(avg_val_rmse_loss)
    train_val_metrics['train_mae'].append(avg_train_mae_loss)
    train_val_metrics['val_mae'].append(avg_val_mae_loss)
    train_val_metrics['train_mape'].append(avg_train_mape_loss)
    train_val_metrics['val_mape'].append(avg_val_mape_loss)

    print(f'Validation Metrics; Epoch {epoch + 1}, Loss (MSE): {avg_val_loss:.4f}, RMSE: {avg_val_rmse_loss:.4f}, MAPE: {avg_val_mape_loss:.4f}, MAE: {avg_val_mae_loss:.4f}')

    chck_folder = os.path.join(f'logs/{job_id}', ckpts)
    chck_save_path = os.path.join(chck_folder, ckpt_file_name)
    os.makedirs(chck_folder, exist_ok=True)

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      save_checkpoint(model, optimizer, epoch, chck_save_path)
      #print(f"New best validation loss: {best_val_loss}, model weights saved.")
  
  return train_val_metrics 
  

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', dest='epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--horizon', dest='horizon', type=int, default=7, help='Timestep horizon')
    parser.add_argument('--size', dest='size', type=str, default='S', help='Dataset size/horizon')
    parser.add_argument('--job_id', dest='job_id', type=str, default='test', help='Slurm job ID')
    parser.add_argument('--device', dest='device', type=str, default='cuda', help='Select device, i.e., "cpu" or "cuda"')
    return parser.parse_args()


if __name__ == '__main__':
  args = get_args()
  train_model(epochs=args.epochs, batch_size=args.batch_size, horizon=args.horizon, size=args.size, job_id=args.job_id, device=args.device)

