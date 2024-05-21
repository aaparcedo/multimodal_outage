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


def train_model(epochs, batch_size, horizon, size, job_id, device):

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
  #model = nn.DataParallel(model).to(device=device)

  transform = transforms.Compose([
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  print(f'device: {device}')

  # Load dataset
  dataset = BlackMarbleDataset(dir_image, size=size, start_index=horizon)

  print(f'size of dataset: {len(dataset)}')

  # Split into train / validation partitions
  n_test = int(len(dataset) * 0.1)
  n_val = int(len(dataset) * 0.2)
  n_train = len(dataset) - (n_val + n_test)
  train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0))

  # Create data loaders
  loader_args = dict(batch_size=batch_size, num_workers=2)
  train_loader = DataLoader(train_set, shuffle=True, **loader_args)
  val_loader = DataLoader(val_set, shuffle=True, **loader_args)
  test_loader = DataLoader(test_set, shuffle=True, **loader_args)

  # Set up optimizer and custom loss function
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  criterion = mse_per_pixel
  
  # Alternative Benchmarks
  rmse = rmse_per_pixel
  mae = mae_per_pixel
  mape = mape_per_pixel

  train_loss_hist = []
  val_loss_hist = []
  
  train_rmse_hist = []
  train_mape_hist = []
  train_mae_hist = []
  
  val_rmse_hist = []
  val_mape_hist = []
  val_mae_hist = []
 
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

    train_loss_hist.append(avg_train_loss)
    train_rmse_hist.append(avg_train_rmse_loss)
    train_mae_hist.append(avg_train_mae_loss)
    train_mape_hist.append(avg_train_mape_loss)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_rmse_loss = val_rmse / len(val_loader)
    avg_val_mae_loss = val_mae / len(val_loader)
    avg_val_mape_loss = val_mape / len(val_loader)

    val_loss_hist.append(avg_val_loss)
    val_rmse_hist.append(avg_val_rmse_loss)
    val_mae_hist.append(avg_val_mae_loss)
    val_mape_hist.append(avg_val_mape_loss)

    print(f'Validation Metrics; Epoch {epoch + 1}, Loss (MSE): {avg_val_loss:.4f}, RMSE: {avg_val_rmse_loss:.4f}, MAPE: {avg_val_mape_loss:.4f}, MAE: {avg_val_mae_loss:.4f}')

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      save_checkpoint(model, optimzer, epoch, f'{job_id}_checkpoint.pth')
      print(f"New best validation loss: {best_val_loss}, model weights saved.")

  save_file_name = f'{args.job_id}_plot.png' 
  save_path = os.path.join('logs', save_file_name)

  plot_training_history(train_loss_hist, val_loss_hist, train_rmse_hist, val_rmse_hist, 
    train_mae_hist, val_mae_hist, train_mape_hist, val_mape_hist, save_path) 

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
      test_rmse += val_rmse_loss.item()
      test_mae += val_mae_loss.item()
      test_mape += val_mape_loss.item()

  avg_test_loss = test_loss / len(test_loader)
  avg_test_rmse_loss = test_rmse / len(test_loader)
  avg_test_mae_loss = test_mae / len(test_loader)
  avg_test_mape_loss = test_mape / len(test_loader)
  
  print(f'Test Results; Loss (MSE): {avg_test_loss:.4f}, RMSE: {avg_test_rmse_loss:.4f}, MAPE: {avg_test_mape_loss:.4f}, MAE: {avg_test_mae_loss:.4f}')

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
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  train_model(epochs=args.epochs, batch_size=args.batch_size, horizon=args.horizon, size=args.size, job_id=args.job_id, device=args.device)

