import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from tqdm import tqdm
from torchvision import transforms
import os

from utils import BlackMarbleDataset, load_adj
from model  import Modified_UNET

dir_image = "/groups/mli/multimodal_outage/data/black_marble/hq/percent_normal/"

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
  if x == 0: 
    x += epsilon
  error = (x - y) / x
  absolute_error = torch.abs(error)
  mean_absolute_error = torch.mean(absolute_error)
  total_mape = 100 * mean_absolute_error
  return total_mape

def train_model(epochs, batch_size, device):

  randomadj = True
  adjdata = "/home/aaparcedo/multimodal_outage/data/graph/adj_mx_fl_k1.csv"
  adjtype = "doubletransition"

  sensor_ids, sensor_id_to_ind, adj_mx = load_adj(adjdata,adjtype)  
  #sensor_ids, sensor_id_to_ind, adj_mx = load_adj(args.adjdata,args.adjtype)
  supports = [torch.tensor(i).to(device) for i in adj_mx]

  #if args.randomadj:
  if randomadj:
    adjinit = None
  else:
    adjinit = supports[0]

  model = Modified_UNET(supports)
  model.to(device=device)

  transform = transforms.Compose([
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  print(f'device: {device}')

  # Load dataset
  # TODO: make start_index a hyperparameter
  dataset = BlackMarbleDataset(dir_image, size='S', start_index=7)

  # Split into train / validation partitions
  n_test = int(len(dataset) * 0.2)
  n_val = int(len(dataset) * 0.2)
  n_train = len(dataset) - (n_val + n_test)
  train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0))


  # Create data loaders
  loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
  train_loader = DataLoader(train_set, shuffle=True, **loader_args)
  val_loader = DataLoader(val_set, shuffle=True, **loader_args)
  test_loader = DataLoader(test_set, shuffle=True, **loader_args)

  # Set up optimizer and custom loss function
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  criterion = mse_per_pixel
  
  # Alternative Benchmarks
  rmse = rmse_per_pixel
  mape = mape_per_pixel

  # Begin training
  for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    train_rmse = 0
    train_mape = 0
    
    with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='day') as pbar:
   
      # item is a tensor of shape [67, 3, 128, 128]
      for item in train_loader:
        past_tensor, future_tensor = (tensor.to(device).permute(0, 2, 1, 3, 4, 5) for tensor in item)

        # Apply transformations
        preds_tensor = model(past_tensor)

        # pixel-wise MSE 
        loss = criterion(preds_tensor, future_tensor)
        
        # pixel-wise RMSE
        train_rmse += rmse(preds_tensor, future_tensor)
        
        #pixel-wise MAPE
        train_mape += mape(preds_tensor, future_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pbar.update(past_tensor.shape[0])    
        epoch_loss += loss.item()
        pbar.set_postfix({'loss (batch)': loss.item()})
      

    model.eval()
    val_loss = 0
    
    with torch.no_grad():
      for item in val_loader:
        past_tensor, future_tensor = (tensor.to(device).permute(0, 2, 1, 3, 4, 5) for tensor in item)

        # Apply transformations
        preds_tensor = model(past_tensor)

        # Pixel-wise MSE
        loss = criterion(preds_tensor, future_tensor)
        val_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    avg_rmse_loss = train_rmse / len(train_loader)
    avg_mape_loss = train_mape / len(train_loader)

    print(f'Epoch {epoch + 1}, \
          Training Loss (MSE): {avg_train_loss:.4f}, \
          RMSE Loss: {avg_rmse_loss:.4f}, \
          MAPE Loss: {avg_mape_loss:.4f}, \
          Validation Loss: {avg_val_loss:.4f}')

  model.eval()
  test_loss = 0

  with torch.no_grad():
    for item in test_loader:
      past_tensor, future_tensor = (tensor.to(device).permute(0, 2, 1, 3, 4, 5) for tensor in item)
      preds_tensor = model(past_tensor)
      loss = criterion(preds_tensor, future_tensor)
      test_loss += loss.item()

  avg_test_loss = test_loss / len(test_loader)
  print(f'Test Loss: {avg_test_loss:.4f}')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
  #args = get_args()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #train_model(model=model, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, device=device, val_percent=args.val / 100)
  train_model(epochs=10, batch_size=4, device=device)

