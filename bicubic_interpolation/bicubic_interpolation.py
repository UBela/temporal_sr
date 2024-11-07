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

def mae_loss(sr_patch, hr_patch):
    return torch.mean(torch.abs(sr_patch - hr_patch))
def mse_loss(sr_patch, hr_patch):
    return torch.mean((sr_patch - hr_patch) ** 2)


def main(config):
    mse = 0.0
    mae = 0.0
    output = []
    hr_images = []
    sr_images = []
    train_data = load_dataset(config.data_path, config.train_start_date, config.train_end_date, config.patch_size)
    test_data = load_dataset(config.data_path, config.test_start_date, config.test_end_date, config.patch_size)
    
    _, train_scale = rescale_data(train_data, custom_scale=None)
    test_data, _ = rescale_data(test_data, custom_scale=train_scale)
    
    test_tensor = torch.stack([torch.tensor(test_data[var].values) for var in ['u10', 'v10']], dim=1)
    #print(test_tensor.shape)
    
    for i in range(test_tensor.shape[0]):
        u10 = test_tensor[i, 0, :, :]
        v10 = test_tensor[i, 1, :, :]
        hr_wind_speed = torch.sqrt(u10 ** 2 + v10 ** 2)
        
        lr_u10 = torch.nn.functional.avg_pool2d(
            u10.unsqueeze(0).unsqueeze(0), 
            kernel_size=config.scaling_factor, 
            stride=config.scaling_factor
        ).squeeze(0)  # Remove the batch dimension but keep channel

        lr_v10 = torch.nn.functional.avg_pool2d(
            v10.unsqueeze(0).unsqueeze(0), 
            kernel_size=config.scaling_factor, 
            stride=config.scaling_factor
        ).squeeze(0) 

        sr_u10 = bicubic_interpolation(lr_u10, config.scaling_factor)
        sr_v10 = bicubic_interpolation(lr_v10, config.scaling_factor)

        sr_wind_speed = torch.sqrt(sr_u10 ** 2 + sr_v10 ** 2)
        
        # Stack sr_u10 and sr_v10 along the channel dimension to store in output
        output.append(torch.stack([sr_u10, sr_v10], dim=1).squeeze(0))
        mse += mse_loss(sr_wind_speed, hr_wind_speed).item()
        mae += mae_loss(sr_wind_speed, hr_wind_speed).item()
        
        hr_images.append(hr_wind_speed.view(-1))
        sr_images.append(sr_wind_speed.view(-1))

    mse /= test_tensor.shape[0]
    mae /= test_tensor.shape[0]
    r2 = r2_score(torch.cat(sr_images), torch.cat(hr_images))
    year = config.test_end_date.split('-')[0]
    print(torch.stack(output, dim=0).shape)
    # Save the output tensor with shape (n, 2, 32, 32)
    torch.save(torch.stack(output), f'{config.output_dir}/baseline_{year}_{config.scaling_factor}x.pt')
    print(f'MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}')


        
    

    
if __name__ == "__main__":
    main(config)