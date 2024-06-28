import torch
from torch.utils.data import DataLoader
from models.unet import Modified_UNET
from utils import BlackMarbleDataset, visualize_risk_map, visualize_results_raster
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import LightningModule, Trainer
from torchmetrics.regression import MeanAbsolutePercentageError, MeanSquaredError, MeanAbsoluteError
import os
import pandas as pd

dir_image = "/groups/mli/multimodal_outage/data/black_marble/hq/original_gap_fill_rectangle"
L.seed_everything(42)

class LitModified_UNET(L.LightningModule):
    def __init__(self, st_gnn, horizon, device):
        super().__init__()
        self.st_gnn = st_gnn
        self.horizon = horizon
        self.model = Modified_UNET(st_gnn=self.st_gnn, horizon=self.horizon, input_channels=1, output_channels=1).to(device=device)
        self.loss_fn = torch.nn.MSELoss()
        self.mean_abs_percentage_error = MeanAbsolutePercentageError()
        self.mean_absolute_error = MeanAbsoluteError()
        self.mean_squared_error = MeanSquaredError()

    def forward(self, x, x_time):
        return self.model(x, x_time)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

test_michael = {'h_michael': pd.Timestamp('2018-10-10')}
test_ian =  {'h_ian': pd.Timestamp('2022-09-26')}
test_idalia = {'h_idalia': pd.Timestamp('2023-08-30')}

def test_model(checkpoint_path, test_dir, st_gnn, horizon, device, batch_size=32):

    # Load the model from checkpoint
  model = LitModified_UNET.load_from_checkpoint(checkpoint_path, st_gnn=st_gnn, horizon=horizon, device=device)
  model.eval()

  # Setup test dataset and dataloader
  test_dataset = BlackMarbleDataset(test_dir, datset_range=30, case_study=test_idalia, horizon=horizon)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
 
  print(f'Test Set: {test_dataset.case_study}')

  # Initialize metrics and storage for predictions and targets
  test_loss, test_mae, test_mape, test_rmse = 0.0, 0.0, 0.0, 0.0
  all_predictions = []
  all_targets = []
  model.to(device)

  with torch.no_grad():
    for batch in test_loader:
      from utils import print_memory_usage
      print_memory_usage()
      x, y, x_time = batch
      x, y = (tensor.to(device).permute(0, 2, 1, 3, 4, 5) for tensor in (x, y))
      yhat = model(x, x_time.to(device))
      loss = model.loss_fn(yhat, y)
      mae = model.mean_absolute_error(yhat, y)
      mape = model.mean_abs_percentage_error(yhat, y)
      rmse = torch.sqrt(model.mean_squared_error(yhat, y))
      test_loss += loss.item()
      test_mae += mae.item()
      test_mape += mape.item()
      test_rmse += rmse.item()

      all_predictions.append(yhat)
      all_targets.append(y)

  n_batches = len(test_loader)
  test_loss /= n_batches
  test_mae /= n_batches
  test_mape /= n_batches
  test_rmse /= n_batches

  all_preds = torch.cat(all_predictions, dim=0)
  all_targets = torch.cat(all_targets, dim=0)

  print(f"Test Metrics:\nTest Loss: {test_loss}\nTest MAE: {test_mae}\nTest MAPE: {test_mape}\nTest RMSE: {test_rmse}")

  return all_preds, all_targets, test_dataset

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Test a trained UNET model checkpoint.")
  parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint.")
  parser.add_argument("--st_gnn", type=str, default='gwnet', help="Whether to use spatio-temporal GNN.")
  parser.add_argument("--horizon", type=int, default=1, help="Horizon value.")
  parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (cuda or cpu).")
  parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing.")

  args = parser.parse_args()

  args.test_dir = "/groups/mli/multimodal_outage/data/black_marble/hq/original_gap_fill_rectangle" 

  all_predictions, all_targets, test_dataset = test_model(
    checkpoint_path=args.checkpoint_path,
    test_dir=dir_image,
    st_gnn=args.st_gnn,
    horizon=args.horizon,
    device=args.device,
    batch_size=args.batch_size
  )

  eid_path = os.path.dirname(os.path.dirname(args.checkpoint_path))  

  preds_path = os.path.join(eid_path, 'preds.pt')
  targets_path = os.path.join(eid_path, 'targets.pt')

  #torch.save(all_predictions, preds_path)
  #torch.save(all_targets, targets_path)
  print(f'Saved preds and targets')

  #visualize_results_raster(preds=all_predictions, save_dir=args.checkpoint_path, save_folder='preds', dataset_dir=dir_image, dataset=test_dataset) 
  #visualize_results_raster(preds=all_targets, save_dir=args.checkpoint_path, save_folder='targets', dataset_dir=dir_image, dataset=test_dataset)  

  visualize_risk_map(all_predictions, args.checkpoint_path, 'preds_risk_maps', test_dataset)
  visualize_risk_map(all_targets, args.checkpoint_path, 'targets_risk_maps', test_dataset)







