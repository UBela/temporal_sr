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
from sklearn.metrics import r2_score
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
NUM_TRAIN_EPOCHS = config.num_train_epochs
LR = config.learning_rate
TRAIN_BATCH_SIZE = config.train_batch_size
EVAL_BATCH_SIZE = config.eval_batch_size
NUM_WORKERS = config.num_workers
PIN_MEMORY = config.pin_memory
N_GPU = config.n_gpu
GAMMA = 0.5
SCALE = config.scaling_factor

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
        epochs_trained = 0
        device = DEVICE
        
        learning_rate = LR
        train_dataset = self.train_dataset
        train_dataloader = self.initialize_dataloader()
        step_size = int(len(train_dataset) / TRAIN_BATCH_SIZE * 200)
        sum_loss = 0
        total_len = 0
        
        if N_GPU > 1:
            self.model = nn.DataParallel(self.model)
            
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=GAMMA)
        
        for epoch in range(epochs_trained, NUM_TRAIN_EPOCHS):
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * (0.1 ** (epoch // int(NUM_TRAIN_EPOCHS * 0.8)))
                
            self.model.train()
            
            with tqdm(total=len(train_dataset) - len(train_dataset) % TRAIN_BATCH_SIZE) as t:
                t.set_description(f'Epoch {epoch}/{NUM_TRAIN_EPOCHS - 1}')
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
    def calc_mse(self, pred, label):
        return torch.mean((pred - label) ** 2)
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
            wind_speed_pred = torch.sqrt(pred_features[:, 0, :, :] ** 2 + pred_features[:, 1, :, :] ** 2)
            wind_speed_label = torch.sqrt(hr_patches[:, 0, :, :] ** 2 + hr_patches[:, 1, :, :] ** 2)
            
            pred_features = torch.cat((wind_speed_pred.unsqueeze(1), pred_features[:, 2:, :, :]), dim=1)
            label_features = torch.cat((wind_speed_label.unsqueeze(1), label_features[:, 2:, :, :]), dim=1)
            
            for i in range(pred_features.shape[1]):  # Iterate over each channel
                pred = pred_features[:, i, :, :].to('cpu').numpy() 
                label = label_features[:, i, :, :].to('cpu').numpy()
                
                # Calculate MSE for the current channel
                feature_mse = self.calc_mse(torch.tensor(pred), torch.tensor(label)).item()
                #
                # print(f'For Feature {i} MSE: {feature_mse}')
                
                total_mse += feature_mse  # Sum up the MSE for the total

        preds_tensor = torch.stack(sr_patches)
        torch.save(preds_tensor, f'{config.preds_dir}/output_{scale}x.pt')

        total_mse /= len(eval_dataloader)
        #r2 = r2_score(all_labels, all_preds) #TODO fix this bug
        r2 = 0
        print(f'MSE: {total_mse:.4f}, R²: {r2:.4f}')

        if epoch == num_train_epochs - 1:
            print(f'Final evaluation done. MSE: {total_mse:.4f}, R²: {r2:.4f}')

        print('Save model')
        self.save_model()


# Main function
if __name__ == '__main__':
    
    # Load and prepare datasets
    # Define dimensions
    num_channels = 6
    lr_height, lr_width = 16, 16
    hr_height, hr_width = 32, 32

    # Generate random input and labels
    # Low-resolution (6x16x16)
   
    # High-resolution (6x32x32)
    hr_sample = torch.rand(num_channels, hr_height, hr_width)
    # Low-resolution (6x16x16)
    lr_sample = nn.AvgPool2d(2)(hr_sample.unsqueeze(0)).squeeze(0)
    # min and max values of the low-resolution and high-resolution samples

    train_dataset = [(lr_sample, hr_sample)]
    eval_dataset = [(lr_sample, hr_sample)]
    lr, hr = train_dataset[0]
    print(lr.shape, hr.shape)
    # Create model instance
    from model import EDSRModel  # Ensure this imports your EDSR model correctly
    model = EDSRModel(in_channels=config.in_channels, feature_channels=config.feature_channels, scaling_factor=config.scaling_factor)
    model = model.to(DEVICE)
    #print model summary
    #print(model)
    # Create Trainer instance
    trainer = CustomTrainer(model=model, args=None, train_dataset=train_dataset, eval_dataset=eval_dataset)

    # Train the model
    train_start_time = time.time()
    trainer.train()
    train_end_time = time.time()
    print("Model training time: {} seconds".format(train_end_time - train_start_time))
