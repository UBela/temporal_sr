from super_image import Trainer
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import os
from torch.optim import Adam, lr_scheduler
from tqdm.auto import tqdm
import yaml
from typing import NamedTuple, Tuple, Union, Optional 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score 

# Load configuration from YAML
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract means and stds from the configuration
means = [config['means'][var] for var in ['u10', 'v10', 'd2m', 't2m', 'msl', 'tp']]
stds = [config['stds'][var] for var in ['u10', 'v10', 'd2m', 't2m', 'msl', 'tp']]

# Constants for the training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_TRAIN_EPOCHS = 100
LR = 1e-4
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 1 
NUM_WORKERS = 2
PIN_MEMORY = True
N_GPU = 1
GAMMA = 0.5
SCALE = 2

class EDSRTrainer(Trainer):
    
    def __init__(self, model, args, train_dataset, eval_dataset=None):
        super().__init__(model, args, train_dataset, eval_dataset)
        
    def initialize_dataloader(self):
        if self.train_dataset is None:
            raise ValueError('train_dataset is not defined.')
        
        data_loader = DataLoader(
            self.train_dataset,
            batch_size=TRAIN_BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY)
        
        return data_loader
    
    def initialize_eval_dataloader(self):
        eval_dataset = self.eval_dataset if self.eval_dataset else self.train_dataset
        data_loader = DataLoader(
            eval_dataset,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
        )
        
        return data_loader

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = False, **kwargs):
        args = self.args
        epochs_trained = 0
        device = DEVICE
        
        num_train_epochs = NUM_TRAIN_EPOCHS
        learning_rate = LR
        train_batch_size = TRAIN_BATCH_SIZE
        train_dataset = self.train_dataset
        train_dataloader = self.initialize_dataloader()
        step_size = int(len(train_dataset) / train_batch_size * 200)
        sum_loss = 0
        total_len = 0
        
        if N_GPU > 1:
            self.model = nn.DataParallel(self.model)
            
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=GAMMA)
        
        for epoch in range(epochs_trained, num_train_epochs):
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * (0.1 ** (epoch // int(num_train_epochs * 0.8)))
                
            self.model.train()
            
            with tqdm(total=len(train_dataset) - len(train_dataset) % train_batch_size) as t:
                t.set_description(f'Epoch {epoch}/{num_train_epochs - 1}')
                for lr_patches, hr_patches in train_dataloader:
                    hr_patches = hr_patches.to(device)
                    lr_patches = lr_patches.to(device)
                    
                    preds = self.model(lr_patches)
                    criterion = nn.L1Loss()
                    
                    loss = criterion(preds, hr_patches)
                    sum_loss += loss.item() * len(lr_patches)
                    total_len += len(lr_patches)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    t.set_postfix(loss=sum_loss / total_len)
                    t.update(len(lr_patches))
            self.eval(epoch)
            
    def denormalize(self, data):
        for i in range(len(means)):
            data[:, i, :, :] = data[:, i, :, :] * stds[i] + means[i]
        return data
        

    def calc_mse(self, pred, hr_patch):
        return torch.mean((pred - hr_patch) ** 2)

    def eval(self, epoch):
        sr_patches = []
        eval_step = 0
        total_mse = 0
        num_train_epochs = NUM_TRAIN_EPOCHS
        scale = self.model.module.config.scale if isinstance(self.model, nn.DataParallel) else SCALE

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

            if eval_step <= 1:
                # Plotting
                fix, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(lr_patches[0, 0, :, :].cpu().numpy(), cmap='inferno')
                ax[0].set_title('LR')
                ax[1].imshow(hr_patches[0, 0, :, :].cpu().numpy(), cmap='inferno')
                ax[1].set_title('HR')
                ax[2].imshow(pred[0, 0, :, :].cpu().numpy(), cmap='inferno')
                plt.savefig(f'output_{epoch}.png')

            pred_features = self.denormalize(pred)            
            label_features = self.denormalize(hr_patches)

            sr_patches.append(pred_features.squeeze(0).to('cpu'))
            wind_speed_features = torch.sqrt(pred_features[:, 0, :, :] ** 2 + pred_features[:, 1, :, :] ** 2)
            wind_speed_labels = torch.sqrt(hr_patches[:, 0, :, :] ** 2 + hr_patches[:, 1, :, :] ** 2)

            pred_features = torch.cat((wind_speed_features.unsqueeze(1), pred_features[:, 2:, :, :]), dim=1)
            label_features = torch.cat((wind_speed_labels.unsqueeze(1), label_features[:, 2:, :, :]), dim=1)

            for i in range(len(pred_features)):
                pred = pred_features[i].to('cpu').numpy() 
                label = label_features[i].to('cpu').numpy()  
                total_mse += self.calc_mse(torch.tensor(pred), torch.tensor(label)).item()  
                all_preds.append(pred)
                all_labels.append(label)

        preds_tensor = torch.stack(sr_patches)
        torch.save(preds_tensor, f'output_{scale}x.pt')

        total_mse /= len(eval_dataloader)
        r2 = r2_score(all_labels, all_preds)

        print(f'MSE: {total_mse:.4f}, R²: {r2:.4f}')

        if epoch == num_train_epochs - 1:
            print(f'Final evaluation done. MSE: {total_mse:.4f}, R²: {r2:.4f}')

        print('Save model')
        self.save_model()
