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



# Extract means and stds from the configuration
means = [config[f'mean_{var}'] for var in ['u10', 'v10', 'd2m', 't2m', 'msl', 'tp']]
stds = [config[f'std_{var}'] for var in ['u10', 'v10', 'd2m', 't2m', 'msl', 'tp']]

class CustomTrainer(Trainer):
    
    def __init__(self, model, args, train_dataset, eval_dataset=None):
        super().__init__(model, args, train_dataset, eval_dataset)
        
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
        sum_loss = 0
        total_len = 0
        
        if n_gpu  > 1:
            self.model = nn.DataParallel(self.model)
            
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=args.gamma)
        
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
                    pred_u_10, _pred_v_10 = preds[:, 0, :, :], preds[:, 1, :, :]
                    label_u_10, label_v_10 = hr_patches[:, 0, :, :], hr_patches[:, 1, :, :]
                    pred_wind_speed = self.get_wind_speed(pred_u_10, _pred_v_10)
                    label_wind_speed = self.get_wind_speed(label_u_10, label_v_10)
                    
                    loss = criterion(pred_wind_speed, label_wind_speed)
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
        for i in range(data.shape[1]):
            data[:, i, :, :] = data[:, i, :, :] * stds[i] + means[i]
        return data
    
    
    def calc_mse(self, pred, label):
        return torch.mean((pred - label) ** 2)
    
    
    def eval(self, epoch):
        args = self.args
        sr_patches = []
        eval_step = 0
        total_mse = 0
        num_train_epochs = config.num_train_epochs
        scale = self.model.module.config.scale if isinstance(self.model, nn.DataParallel) else config.scaling_factor

        device = DEVICE
        eval_dataloader = self.initialize_eval_dataloader()

        self.model.eval()
        
        all_preds = []
        all_labels = []

        for lr_patches, hr_patches in eval_dataloader:
            #get max of the lr_patches and hr_patches
           
            hr_patches = hr_patches.to(device)
            lr_patches = lr_patches.to(device)
            eval_step += 1
            
            with torch.no_grad():
                pred = self.model(lr_patches)

            if eval_step <= 1:
                # Plotting
                fix, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(lr_patches[0, 0, :, :].cpu().numpy(), cmap='inferno')
                ax[0].set_title('Low resolution Input')
                ax[1].imshow(hr_patches[0, 0, :, :].cpu().numpy(), cmap='inferno')
                ax[1].set_title('High resolution Label')
                ax[2].imshow(pred[0, 0, :, :].cpu().numpy(), cmap='inferno')
                ax[2].set_title('Super resolution Output')
                
                plt.savefig(f'{config.output_dir}/output_epoch_{epoch}.png')

            pred_features = self.denormalize(pred)            
            label_features = self.denormalize(hr_patches)

            sr_patches.append(pred_features.squeeze(0).to('cpu'))
            wind_speed_pred = self.get_wind_speed(pred_features[:, 0, :, :], pred_features[:, 1, :, :])
            wind_speed_label = self.get_wind_speed(label_features[:, 0, :, :], label_features[:, 1, :, :])
            
            all_preds.append(wind_speed_pred.view(-1))
            all_labels.append(wind_speed_label.view(-1))
            
            
            total_mse += self.calc_mse(wind_speed_pred.to('cpu'), wind_speed_label.to('cpu')).item()
        # get highest value of pred_features
        
        #print(pred_features)
        preds_tensor = torch.stack(sr_patches)
        torch.save(preds_tensor, f'{config.preds_dir}/output_{scale}x.pt')

        total_mse /= len(eval_dataloader)
        #to tensor
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        r2 = r2_score(all_labels, all_preds).item()
        print(f'MSE: {total_mse:.4f}, R²: {r2:.4f}')

        if epoch == num_train_epochs - 1:
            print(f'Final evaluation done. MSE: {total_mse:.4f}, R²: {r2:.4f}')

        print('Save model')
        self.save_model(output_dir=config.model_path)


if __name__ == '__main__':
    
    num_channels = 6
    hr_height, hr_width = 32, 32

    train_dataset = []
    eval_dataset = []
    
    for _ in range(10):
        hr_sample = torch.rand(num_channels, hr_height, hr_width)
        lr_sample = nn.AvgPool2d(2)(hr_sample.unsqueeze(0)).squeeze(0)
        hr_sample = hr_sample[:2,:,:] # only u and v component
        train_dataset.append((lr_sample, hr_sample))
        eval_dataset.append((lr_sample, hr_sample))
    
    from model import EDSRModel  
    model = EDSRModel(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        feature_channels=config.feature_channels,
        scaling_factor=config.scaling_factor
    )
    model = model.to(DEVICE)
    
    trainer = CustomTrainer(model=model, args=None, train_dataset=train_dataset, eval_dataset=eval_dataset)

    train_start_time = time.time()
    trainer.train()
    train_end_time = time.time()
    print("Model training time: {} seconds".format(train_end_time - train_start_time))
    
    
