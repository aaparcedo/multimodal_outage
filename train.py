import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchvision import transforms
import os


from graph_unet.dataset_loader import BlackMarbleDataset
from graph_unet.Graph_Unet import Modified_UNET

dir_image = "/groups/mli/multimodal_outage/data/black_marble/hq/percent_normal/"

def mse_per_pixel(x, y):
    # Calculate squared difference
    squared_diff = (x - y) ** 2
    
    # Compute mean across all dimensions except for the batch dimension
    # This averages the MSE across all pixels in each image in the batch
    mse_per_image = torch.mean(squared_diff, dim=[1, 2, 3])
    
    # Finally, compute the average across all images in the batch
    total_mse = torch.mean(mse_per_image)
    return total_mse


def train_model(model, epochs, batch_size, device):

  transform = transforms.Compose([
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  # Load dataset
  dataset = BlackMarbleDataset(dir_image)

  # Split into train / validation partitions
  n_val = int(len(dataset) * 0.3)
  n_train = len(dataset) - n_val
  train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

  # Create data loaders
  loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
  train_loader = DataLoader(train_set, shuffle=True, **loader_args)
  val_loader = DataLoader(val_set, shuffle=False, **loader_args)

  # Set up optimizer and custom loss function
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  criterion = mse_per_pixel

  # Begin training
  for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='day') as pbar:
   
      # item is a tensor of shape [67, 3, 128, 128]
      for item in train_loader:
        images = item.to(device).squeeze(0)
        
        # Apply transformations
        image_preds = model(images)

        # pixel-wise MSE 
        loss = criterion(image_preds, images)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pbar.update(item.shape[0])    
        epoch_loss += loss.item()
        pbar.set_postfix({'loss (batch)': loss.item()})
     
        # missing validation
        # missing logs

    torch.cuda.empty_cache()

# forward pass should be something like:

# unet (contract)
# encoder

# images.shape => [B, S, N, F, T] ~ [1, 4, 67, 9, 12] 
# one batch, 4 days/samples per batch, 67 counties, 1 (eagle-i) + 8 (image embed), 12 timesteps in the past
# image_embed_preds = gwn(images)

# decoder shape: [1, 4, 67, 8, 12x8]
# image_decoded = decoder(image_embed_preds)

# unet expand shape: [1, 4, 67, 128x128, 12]
# 4 images, 67 counties, 128x128 image size, 12 timesteps into the future
# image_preds = unet_expand(image_decoded)


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
  model = Modified_UNET()
  model.to(device=device)
  #train_model(model=model, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, device=device, val_percent=args.val / 100)
  train_model(model=model, epochs=5, batch_size=1, device=device)

