import numpy as np
import cv2
from box import Box
import argparse
import yaml
import torch
from torcheval.metrics.functional import r2_score 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from process_data import load_dataset, rescale_data


def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

parser = argparse.ArgumentParser(description='Run model training with configuration.')
parser.add_argument('config_path', type=str, help='Path to the configuration yaml file.')
args = parser.parse_args()

# Load and box the configuration
config = Box(load_config(args.config_path)['TrainingConfig'])

def bicubic_interpolation(lr_patch, scaling_factor):
    lr_patch = lr_patch.permute(1, 2, 0).numpy()
    sr_patch = cv2.resize(lr_patch, 
                        dsize=(lr_patch.shape[1] * scaling_factor, 
                        lr_patch.shape[0] * scaling_factor), 
                        interpolation=cv2.INTER_CUBIC)
    sr_patch = torch.tensor(sr_patch).unsqueeze(2).permute(2, 0, 1)
    return sr_patch
def bicubic_interpolation(lr_patch, scaling_factor):
    if lr_patch.dim() == 3:
        lr_patch = lr_patch.unsqueeze(0) 
    sr_patch = torch.nn.functional.interpolate(
        lr_patch, 
        scale_factor=scaling_factor, 
        mode='bicubic', 
        align_corners=False
    )
    return sr_patch.squeeze(0)  

def mae_loss(sr_patch, hr_patch):
    return torch.mean(torch.abs(sr_patch - hr_patch))
def mse_loss(sr_patch, hr_patch):
    return torch.mean((sr_patch - hr_patch) ** 2)


def main(config, test_year):
    mse = 0.0
    mae = 0.0
    output = []
    hr_images = []
    sr_images = []
    test_start_date = f'{test_year}-01-01'
    test_end_date = f'{test_year}-12-31'
    train_data = load_dataset(config.data_path, config.train_start_date, config.train_end_date, config.patch_size)
    test_data = load_dataset(config.data_path, test_start_date, test_end_date, config.patch_size)
    
    _, train_scale = rescale_data(train_data, custom_scale=None)
    test_data, _ = rescale_data(test_data, custom_scale=train_scale)
    
    test_tensor = torch.stack([torch.tensor(test_data[var].values) for var in ['u10', 'v10', 't2m']], dim=1)
    print(f'Test data for {test_year}')
    for i in range(test_tensor.shape[0]):
    
        lr_vars = torch.nn.functional.avg_pool2d(
            test_tensor[i,:,:,:].unsqueeze(0), 
            kernel_size=config.scaling_factor, 
            stride=config.scaling_factor
        ).squeeze(0)
    
        sr_vars = bicubic_interpolation(lr_vars, config.scaling_factor)
    
        output.append(sr_vars.squeeze(0))
        mse += mse_loss(sr_vars, test_tensor[i,:,:,:]).item()
        mae += mae_loss(sr_vars, test_tensor[i,:,:,:]).item()
        hr_images.append(test_tensor[i,:,:,:].view(-1))
        sr_images.append(sr_vars.view(-1))
    mse /= test_tensor.shape[0]
    mae /= test_tensor.shape[0]
    r2 = r2_score(torch.cat(sr_images), torch.cat(hr_images))
    torch.save(torch.stack(output), f'{config.output_dir}/baseline_{test_year}_{config.scaling_factor}x.pt')
    print(f'MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}')


if __name__ == "__main__":
    for year in range(1984, 2024):
        main(config, year)
    