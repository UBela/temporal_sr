from super_image import Trainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from tqdm.auto import tqdm
import yaml
import argparse
from box import Box
import time
import matplotlib.pyplot as plt
from torcheval.metrics.functional import r2_score 
from typing import Optional, Union
import numpy as np


def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

parser = argparse.ArgumentParser(description='Run model training with configuration.')
parser.add_argument('config_path', type=str, help='Path to the configuration yaml file.')
args = parser.parse_args()

# Load and box the configuration
config = Box(load_config(args.config_path)['TrainingConfig'])

# Constants
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
# Extract means and stds from the configuration
means = [config.mean_u10, config.mean_v10, config.mean_d2m, config.mean_t2m, config.mean_msl, config.mean_tp]
stds = [config.std_u10, config.std_v10, config.std_d2m, config.std_t2m, config.std_msl, config.std_tp]

class CustomTrainer(Trainer):
    
    def __init__(self, model, args, train_dataset, eval_dataset=None):
        super().__init__(model, args, train_dataset, eval_dataset)
        self.train_losses = []
        self.eval_mses = []
        self.eval_maes = []
        self.eval_r2 = []
        self.best_eval_mse = float('inf')
        self.best_eval_mae = float('inf')
        self.best_eval_r2 = 0.0
    def initialize_dataloader(self):
        if self.train_dataset is None:
            raise ValueError('train_dataset is not defined.')
        
        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size, 
            shuffle=False, 
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory)
        
        return data_loader
    
    def get_wind_speed(self, u , v):
        
        return torch.sqrt(u ** 2 + v ** 2)
    
    
    def initialize_eval_dataloader(self):
        eval_dataset = self.eval_dataset if self.eval_dataset else self.train_dataset
        data_loader = DataLoader(
            eval_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
        )
        
        return data_loader

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = False, **kwargs):
        
        args = self.args
        epochs_trained = 0
        device = args.device

        learning_rate = args.learning_rate
        num_train_epochs = args.num_train_epochs
        train_dataset = self.train_dataset
        train_dataloader = self.initialize_dataloader()
        train_batch_size = args.train_batch_size
        step_size = int(len(train_dataset) / train_batch_size * 200)
        n_gpu = args.n_gpu
        
        
        if n_gpu  > 1:
            self.model = nn.DataParallel(self.model)
            
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=self.args.gamma)
        
        for epoch in range(epochs_trained, num_train_epochs):
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * (0.1 ** (epoch // int(num_train_epochs * 0.8)))
                
            self.model.train()
            epoch_losses = AverageMeter()
            
            with tqdm(total=len(train_dataset) - len(train_dataset) % train_batch_size) as t:
                t.set_description(f'Epoch {epoch}/{num_train_epochs - 1}')
                for lr_patches, hr_patches in train_dataloader:
                    hr_patches = hr_patches.to(device)
                    lr_patches = lr_patches.to(device)
                    
                    preds = self.model(lr_patches)
                    criterion = nn.L1Loss()
                    
                    loss = criterion(preds, hr_patches)
                    epoch_losses.update(loss.item(), len(lr_patches))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    t.set_postfix(loss=f'{epoch_losses.avg:.6}')
                    t.update(len(lr_patches))
                    
                self.train_losses.append(epoch_losses.avg)   
                eval_mae, eval_mse, eval_r2 = self.eval(epoch) 
                
                self.eval_maes.append(eval_mae)
                self.eval_mses.append(eval_mse)
                
                if (eval_mae < self.best_eval_mae or
                    eval_mse < self.best_eval_mse or
                    eval_r2 > self.best_eval_r2):
                    print("Improvement detected. Saving model...")
                    self.save_model(output_dir=config.model_path)
                    self.best_eval_mae = min(eval_mae, self.best_eval_mae)
                    self.best_eval_mse = min(eval_mse, self.best_eval_mse)
                    self.best_eval_r2 = max(eval_r2, self.best_eval_r2)
                
                self.eval_maes.append(eval_mae)
                self.eval_mses.append(eval_mse)
                self.eval_r2s.append(eval_r2)
                

    def denormalize(self, data):
        for i in range(data.shape[1]):
            data[:, i, :, :] += means[i]
        return data
    
    
    def calc_mse(self, pred, label):
        return torch.mean((pred - label) ** 2)
    
    def calc_mae(self, pred, label):
        return torch.mean(torch.abs(pred - label))
    
    
    def eval(self, epoch):
        args = self.args
        sr_patches = []
        eval_step = 0
        total_mse = 0
        total_mae = 0
        num_train_epochs = config.num_train_epochs
        scale = self.model.module.config.scale if isinstance(self.model, nn.DataParallel) else config.scaling_factor

        device = DEVICE
        eval_dataloader = self.initialize_eval_dataloader()

        self.model.eval()
        
        all_preds = []
        all_labels = []

        for lr_patches, hr_patches in eval_dataloader:
        
            hr_patches = hr_patches.to(device)
            lr_patches = lr_patches.to(device)
            eval_step += 1
            
            with torch.no_grad():
                pred = self.model(lr_patches)
            pred_features = self.denormalize(pred)            
            label_features = self.denormalize(hr_patches)
            if eval_step <= 1:
               
                # Plotting
                fix, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(lr_patches[0, 0, :, :].cpu().numpy(), cmap='inferno')
                ax[0].set_title('Low-resolution Input')
                ax[1].imshow(label_features[0, 0, :, :].cpu().numpy(), cmap='inferno')
                ax[1].set_title('High-resolution Label')
                ax[2].imshow(pred_features[0, 0, :, :].cpu().numpy(), cmap='inferno')
                ax[2].set_title('Super-resolution Output')
                
                plt.savefig(f'{config.output_dir}/output_epoch_{epoch}.png')
                plt.close(fix)
            

            sr_patches.append(pred_features.squeeze(0).to('cpu'))
            wind_speed_pred = self.get_wind_speed(pred_features[:, 0, :, :], pred_features[:, 1, :, :])
            wind_speed_label = self.get_wind_speed(label_features[:, 0, :, :], label_features[:, 1, :, :])
            
            all_preds.append(wind_speed_pred.view(-1))
            all_labels.append(wind_speed_label.view(-1))
            
            
            total_mse += self.calc_mse(wind_speed_pred.to('cpu'), wind_speed_label.to('cpu')).item()
            total_mae += self.calc_mae(wind_speed_pred.to('cpu'), wind_speed_label.to('cpu')).item()
        preds_tensor = torch.stack(sr_patches)
        year = config.test_end_date.split('-')[0]
        torch.save(preds_tensor, f'{config.preds_dir}/EDSR_{year}_{scale}x.pt')

        total_mse /= len(eval_dataloader)
        total_mae /= len(eval_dataloader)
        #to tensor
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        r2 = r2_score(all_labels, all_preds).item()
        print(f'MSE: {total_mse:.6f}, R²: {r2:.6f}, MAE: {total_mae:.6f}')

        if epoch == num_train_epochs - 1:
            print(f'Final evaluation done. MSE: {total_mse:.6f}, R²: {r2:.6f}, MAE: {total_mae:.6f}')

        return total_mae, total_mse, r2
    
    def plot_metrics(self):
        epochs = range(len(self.train_losses))

        train_losses_log = torch.log(torch.tensor(self.train_losses))
        eval_mses_log = torch.log(torch.tensor(self.eval_mses))
        eval_maes_log = torch.log(torch.tensor(self.eval_maes))

        plt.figure(figsize=(18, 5))

        # Plot training loss
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_losses_log, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Log L1 Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()

        # Plot evaluation MSE
        plt.subplot(1, 3, 2)
        plt.plot(epochs, eval_mses_log, label="Evaluation MSE", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Log MSE")
        plt.title("Evaluation MSE Over Epochs")
        plt.legend()

        # Plot evaluation MAE
        plt.subplot(1, 3, 3)
        plt.plot(epochs, eval_maes_log, label="Evaluation MAE", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Log MAE")
        plt.title("Evaluation MAE Over Epochs")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{config.output_dir}/training_eval_metrics_{config.scaling_factor}x.png')
        plt.close()
