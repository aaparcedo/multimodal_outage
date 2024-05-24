from utils import BlackMarbleDataset, plot_training_history, plot_error_metrics
from train import train_model
from test import test_model
from torchvision import transforms
import argparse
import pandas as pd
import os
import numpy as np

dir_image = "/groups/mli/multimodal_outage/data/black_marble/hq/percent_normal/"

def run_experiment(epochs, batch_size, horizon, size, job_id, num_runs, device):

  transform = transforms.Compose([
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  train_ia_id, test_m = {'h_ian': pd.Timestamp('2022-09-26'), 'h_idalia': pd.Timestamp('2023-08-30')}, {'h_michael': pd.Timestamp('2018-10-10')}
  train_m_id, test_ia = {'h_michael': pd.Timestamp('2018-10-10'), 'h_idalia': pd.Timestamp('2023-08-30')}, {'h_ian': pd.Timestamp('2022-09-26')}
  train_ia_m, test_id = {'h_ian': pd.Timestamp('2022-09-26'), 'h_michael': pd.Timestamp('2018-10-10')}, {'h_idalia': pd.Timestamp('2023-08-30')}

  case_study_events = [(train_ia_id, test_m), (train_m_id, test_ia), (train_ia_m, test_id)]

  # val metrics
  runs_metrics = {}
  
  
  overall_test_metrics = {'loss': [], 'rmse': [], 'mae': [], 'mape': []}

  for case_train, case_test in case_study_events:
 
    train_dataset = BlackMarbleDataset(dir_image, size=size, case_study=case_train, start_index=horizon, transform=transform)
    test_dataset = BlackMarbleDataset(dir_image, size=size, case_study=case_test, start_index=horizon, transform=transform)

    print(f'Train events: {case_train.keys()}, Test event: {case_test.keys()}')
 
    train_h_names = list(case_train.keys())
    test_h_name = list(case_test.keys())

    run_test_loss_hist = []
    run_test_rmse_hist = []
    run_test_mae_hist = []
    run_test_mape_hist = []

    run_val_loss_hist = []
    run_val_rmse_hist = []
    run_val_mae_hist = []
    run_val_mape_hist = []

    # Repeat the same experiment 3 times
    for run in range(num_runs):

      ckpt_run_save_file_name = f'trained_on_{train_h_names[0]}_{train_h_names[1]}_run{run}_ckpt.pth'

      train_metrics = train_model(epochs=epochs, batch_size=batch_size, horizon=horizon, job_id=job_id, ckpt_file_name=ckpt_run_save_file_name, device=device, dataset=train_dataset)
      test_metrics = test_model(epochs=epochs, batch_size=batch_size, horizon=horizon, job_id=job_id, ckpt_file_name=ckpt_run_save_file_name, device=device, dataset=test_dataset)

      save_file_name = f'trained_on_{train_h_names[0]}_{train_h_names[1]}_run{run}_plot.png'
      save_folder_path = os.path.join(f'logs/{job_id}', 'figs')
      run_save_path = os.path.join(save_folder_path, save_file_name)
      os.makedirs(save_folder_path, exist_ok=True)

      plot_training_history(train_metrics['train_loss'], train_metrics['val_loss'], \
        train_metrics['train_rmse'], train_metrics['val_rmse'], \
        train_metrics['train_mae'], train_metrics['val_mae'], \
        train_metrics['train_mape'], train_metrics['val_mape'], run_save_path)
      
      run_val_loss_hist.append(train_metrics['val_loss'])
      run_val_rmse_hist.append(train_metrics['val_rmse'])
      run_val_mae_hist.append(train_metrics['val_mae'])
      run_val_mape_hist.append(train_metrics['val_mape'])

      run_test_loss_hist.append(test_metrics['loss'])
      run_test_rmse_hist.append(test_metrics['rmse'])
      run_test_mae_hist.append(test_metrics['mae'])
      run_test_mape_hist.append(test_metrics['mape'])

      print(f'Run number: {run + 1}')
      print(train_metrics)
      print(test_metrics)
      print(f"Test Loss: {test_metrics['loss']:.4f}, RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, MAPE: {test_metrics['mape']:.4f}")
      print(f"\n=========================================================================================================================================\n")

    run_avg_test_loss = sum(run_test_loss_hist) / len(run_test_loss_hist) 
    run_avg_test_rmse = sum(run_test_rmse_hist) / len(run_test_rmse_hist)
    run_avg_test_mae = sum(run_test_mae_hist) / len(run_test_mae_hist)
    run_avg_test_mape = sum(run_test_mape_hist) / len(run_test_mape_hist)

    overall_test_metrics['loss'].append(run_avg_test_loss)
    overall_test_metrics['rmse'].append(run_avg_test_mae)
    overall_test_metrics['mae'].append(run_avg_test_mae)
    overall_test_metrics['mape'].append(run_avg_test_mape)

    if test_h_name[0] not in runs_metrics:
        runs_metrics[test_h_name[0]] = {}

    runs_metrics[test_h_name[0]]['val_loss'] = run_val_loss_hist
    runs_metrics[test_h_name[0]]['val_rmse'] = run_val_rmse_hist
    runs_metrics[test_h_name[0]]['val_mae'] = run_val_mae_hist
    runs_metrics[test_h_name[0]]['val_mape'] = run_val_mape_hist

    print(f'Average over {num_runs} runs: ')
    print(f'Loss={run_avg_test_loss}, RMSE={run_avg_test_rmse}, MAE={run_avg_test_mae}, MAPE: {run_avg_test_mape}')
    print(f"\n\n=========================================================================================================================================")
    print(f"=========================================================================================================================================\n\n")
    

    
    experiment_file_name = 'case_study_metrics_over_runs.png'
    experiment_save_path = os.path.join(save_folder_path, experiment_file_name)

    # Plot the average of the num_runs here
    plot_error_metrics(runs_metrics, experiment_save_path)
  
  avg_loss = np.mean(overall_test_metrics['loss'])
  avg_rmse = np.mean(overall_test_metrics['rmse'])
  avg_mae = np.mean(overall_test_metrics['mae'])
  avg_mape = np.mean(overall_test_metrics['mape'])
  
  print(f'Average case study test error over {args.num_runs} runs:')
  print(f'Loss: {avg_loss}, RMSE: {avg_rmse}, MAE: {avg_mae}, MAPE: {avg_mape}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', dest='epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--horizon', dest='horizon', type=int, default=7, help='Timestep horizon')
    parser.add_argument('--size', dest='size', type=str, default='S', help='Dataset size/horizon')
    parser.add_argument('--job_id', dest='job_id', type=str, default='test', help='Slurm job ID')
    parser.add_argument('--num_runs', dest='num_runs', type=int, default=1, help='Number of times to repeat the same experiment')
    parser.add_argument('--device', dest='device', type=str, default='cuda', help='Select device, i.e., "cpu" or "cuda"')
    return parser.parse_args()


if __name__ == '__main__':
  args = get_args()
  print(args)
  run_experiment(epochs=args.epochs, batch_size=args.batch_size, horizon=args.horizon, size=args.size, job_id=args.job_id, num_runs=args.num_runs, device=args.device)