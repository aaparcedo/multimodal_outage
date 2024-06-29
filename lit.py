import lightning as L
from models.unet import Modified_UNET
import torch
import torch.nn as nn
from utils import BlackMarbleDataset, print_memory_usage, prepare_dataloaders, visualize_risk_map, visualize_results_raster
import pandas as pd
import argparse
from torch.optim import lr_scheduler
from torchmetrics.regression import MeanAbsolutePercentageError, MeanSquaredError, MeanAbsoluteError
import os 
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback

#L.seed_everything(42)

DATA_PATH  = "/groups/mli/multimodal_outage/data/black_marble/hq/original_gap_fill_rectangle_proximity"

class LitModified_UNET(L.LightningModule):
  def __init__(self, st_gnn, horizon, device):
    super().__init__()
    self.st_gnn = st_gnn
    self.horizon = horizon
    self.model = Modified_UNET(st_gnn=self.st_gnn, horizon=self.horizon, input_channels=1, output_channels=1).to(device=device)
    self.loss_fn = nn.MSELoss()
    self.mean_abs_percentage_error = MeanAbsolutePercentageError()
    self.mean_absolute_error = MeanAbsoluteError()
    self.mean_squared_error = MeanSquaredError()

  def training_step(self, batch):
    x, y, x_time = batch
    x, y = (tensor.to(self.device).permute(0, 2, 1, 3, 4, 5) for tensor in (x, y))
    yhat = self.model(x, x_time.to(self.device))
    loss = self.loss_fn(yhat, y)
    with torch.no_grad():
      mae = self.mean_absolute_error(yhat, y)
      mape = self.mean_abs_percentage_error(yhat, y)
      rmse = torch.sqrt(self.mean_squared_error(yhat, y))
    self.log('train_loss', loss, prog_bar=True)
    self.log('train_mae', mae)
    self.log('train_mape', mape)
    self.log('train_rmse', rmse)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y, x_time = batch
    x, y = (tensor.to(self.device).permute(0, 2, 1, 3, 4, 5) for tensor in (x, y))
    yhat = self.model(x, x_time.to(self.device))
    loss = self.loss_fn(yhat, y)
    rmse = torch.sqrt(self.mean_squared_error(yhat, y))
    mae = self.mean_absolute_error(yhat, y)
    mape = self.mean_abs_percentage_error(yhat, y)
    self.log('val_loss', loss, prog_bar=True)
    self.log('val_mae', mae)
    self.log('val_mape', mape)
    self.log('val_rmse', rmse)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10) 
    print(optimizer)
    print(scheduler)
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
          'scheduler': scheduler,
          'monitor': 'val_loss',
        'interval': 'epoch',
        'frequency': 1
      }
    }


def main(st_gnn, test_case, epochs, batch_size, horizon, dataset_range, job_id, num_runs, device):

  early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=True,

    mode='min'
  )

  checkpoint_save_path = f'/home/aaparcedo/multimodal_outage/logs/{job_id}/checkpoints/'
  os.makedirs(checkpoint_save_path, exist_ok=True)

  checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath = checkpoint_save_path, 
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min'
  )

  train_loader, train_set, val_loader, val_set, test_loader, test_dataset = prepare_dataloaders(test_case, batch_size,  horizon)

  test_best_model_callback = TestBestModelCallback(test_loader, device, test_dataset)
  print_metrics_callback = PrintMetricsCallback(val_loader, device)

  wandb_logger = WandbLogger(tags=['rand_seed'], log_model="all", project="mst_gnn", id=job_id)

  model = LitModified_UNET(st_gnn=st_gnn, horizon=horizon, device=device)
  trainer = L.Trainer(callbacks=[early_stop_callback, checkpoint_callback, print_metrics_callback, test_best_model_callback], max_epochs=epochs, logger=wandb_logger)
  trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

class TestBestModelCallback(Callback):
  def __init__(self, test_loader, device, test_dataset):
    self.test_loader = test_loader
    self.device = device
    self.test_dataset = test_dataset
    self.preds = []
    self.targets = []

  def on_train_end(self, trainer, pl_module):
    # Load the best model checkpoint
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = LitModified_UNET.load_from_checkpoint(best_model_path, st_gnn=pl_module.st_gnn, horizon=pl_module.horizon, device=self.device)
    best_model.eval()

    test_loss, test_mae, test_mape, test_rmse = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
      for batch in self.test_loader:
        x, y, x_time = batch
        x, y = (tensor.to(self.device).permute(0, 2, 1, 3, 4, 5) for tensor in (x, y))
        yhat = best_model.model(x, x_time.to(self.device))
        loss = pl_module.loss_fn(yhat, y)
        mae = pl_module.mean_absolute_error(yhat, y)
        mape = pl_module.mean_abs_percentage_error(yhat, y)
        rmse = torch.sqrt(pl_module.mean_squared_error(yhat, y))
        test_loss += loss.item()
        test_mae += mae.item()
        test_mape += mape.item()
        test_rmse += rmse.item()

        self.preds.append(yhat)
        self.targets.append(y)

    # torch.Size([58, 67, 1, 1, 128, 128])
    self.preds = torch.cat(self.preds, dim=0)
    self.targets = torch.cat(self.targets, dim=0)
    
    n_batches = len(self.test_loader)
    test_loss /= n_batches
    test_mae /= n_batches
    test_mape /= n_batches
    test_rmse /= n_batches

    print(f"Best Model Metrics:\nTest Loss: {test_loss}; Test MAE: {test_mae}; Test MAPE: {test_mape}; Test RMSE: {test_rmse}")

    dates, preds_np = visualize_risk_map(self.preds, 'test', 'preds_risk_maps', self.test_dataset)
    _, targets_np = visualize_risk_map(self.targets, 'test', 'targets_risk_maps', self.test_dataset)

    data = []
    for date, pred_image, target_image in zip(dates, preds_np, targets_np):
      data.append([date, wandb.Image(pred_image), wandb.Image(target_image)])

    columns = ["Date", "Prediction", "Target"]
    table = wandb.Table(columns=columns, data=data)
    wandb.log({"predictions_table": table})
  
class PrintMetricsCallback(Callback):
  def __init__(self, val_loader, device):
    self.val_loader = val_loader
    self.device = device

  def on_train_end(self, trainer, pl_module):
    # Load the best model checkpoint
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = LitModified_UNET.load_from_checkpoint(best_model_path, st_gnn=pl_module.st_gnn, horizon=pl_module.horizon, device=self.device)
    best_model.eval()

    val_loss, val_mae, val_mape, val_rmse = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
      for batch in self.val_loader:
        x, y, x_time = batch
        x, y = (tensor.to(self.device).permute(0, 2, 1, 3, 4, 5) for tensor in (x, y))
        yhat = best_model.model(x, x_time.to(self.device))
        loss = pl_module.loss_fn(yhat, y)
        mae = pl_module.mean_absolute_error(yhat, y)
        mape = pl_module.mean_abs_percentage_error(yhat, y)
        rmse = torch.sqrt(pl_module.mean_squared_error(yhat, y))
        val_loss += loss.item()
        val_mae += mae.item()
        val_mape += mape.item()
        val_rmse += rmse.item()

    n_batches = len(self.val_loader)
    val_loss /= n_batches
    val_mae /= n_batches
    val_mape /= n_batches
    val_rmse /= n_batches

    print(f"Best Model Metrics:\nValidation Loss: {val_loss}\nValidation MAE: {val_mae}\nValidation MAPE: {val_mape}\nValidation RMSE: {val_rmse}")

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--st_gnn', dest='st_gnn', type=str, default='gwnet', help='Pick a st-gnn to train/test.')
  parser.add_argument('--epochs', dest='epochs', type=int, default=5, help='Number of epochs')
  parser.add_argument('--case', dest='case', type=str, default='michael', help='Test case')
  parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='Batch size')
  parser.add_argument('--horizon', dest='horizon', type=int, default=7, help='Timestep horizon')
  parser.add_argument('--dataset_range', dest='dataset_range', type=int, default=30, help='Dataset size/range')
  parser.add_argument('--job_id', dest='job_id', type=str, default='test', help='Slurm job ID')
  parser.add_argument('--num_runs', dest='num_runs', type=int, default=1, help='Number of times to repeat the same experiment')
  parser.add_argument('--device', dest='device', type=str, default='cuda', help='Select device, i.e., "cpu" or "cuda"')
  return parser.parse_args()

if __name__ == '__main__':
  args = get_args()
  print(args)
  main(st_gnn=args.st_gnn, test_case=args.case, epochs=args.epochs, batch_size=args.batch_size, horizon=args.horizon, dataset_range=args.dataset_range, job_id=args.job_id, num_runs=args.num_runs, device=args.device)
