import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
import os
from PIL import Image
import yaml
from box import Box
import argparse

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

parser = argparse.ArgumentParser(description='Run model training with config')
parser.add_argument('config_path', type=str, help='Path to the config file')

args = parser.parse_args()
config = Box(load_config(args.config_path)['TrainingConfig'])

# Use config values instead of hardcoding
DATA_PATH = config.data_path
SCALE = config.scaling_factor
PATCH_SIZE = config.patch_size
FEATURE_LIST = config.feature_list
TRAIN_START = config.train_start_date
TRAIN_END = config.train_end_date
TEST_START = config.test_start_date
TEST_END = config.test_end_date
means = [config.mean_u10, config.mean_v10, config.mean_d2m, config.mean_t2m, config.mean_msl, config.mean_tp]
def load_dataset(data, start, end, patch_size):
    dataset = xr.open_dataset(data).sel(valid_time=slice(start, end)).isel(
        longitude=slice(0, patch_size), latitude=slice(0, patch_size))
    return dataset

def rescale_data(data, custom_scale = None):
    returned_scale = {}
    for var in data:
        
        if custom_scale and var in custom_scale:
            min_val = custom_scale[var]['min']
            max_val = custom_scale[var]['max']
        else:
            min_val = data[var].values.min()
            max_val = data[var].values.max()
            
            returned_scale[var] = {'min': min_val, 'max': max_val}
            
            
        # Rescale to the target range [0, 1]
        data[var] = (data[var] - min_val) / (max_val - min_val)  
        
        
    return data, returned_scale


def save_means_stds(config_path, dataset, feature_names):
    # Stack all data for mean and std calculation
    feature_tensors = []
    for i in range(len(dataset)):
        lr_patches, _ = dataset[i]  
        feature_tensors.append(lr_patches)
    
    features = torch.stack(feature_tensors, dim=0)  # Shape: (num_samples, num_channels, H, W)
    means = features.view(features.shape[1], -1).mean(dim=1)  # Mean across spatial dimensions
    stds = features.view(features.shape[1], -1).std(dim=1)  # Std across spatial dimensions
    
    # Load the existing configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file) or {}
    training_config = config.get("TrainingConfig", {})

    # Save means and stds with feature names
    for mean, std, var in zip(means, stds, feature_names):
        training_config[f'mean_{var}'] = float(mean)
        training_config[f'std_{var}'] = float(std)
    
    config["TrainingConfig"] = training_config

    # Write updated configuration to file
    with open(config_path, 'w') as file:
        yaml.dump(config, file)


def average_pooling(data, scale, to_int=False):
    new_h, new_w = data.shape[0] // scale, data.shape[1] // scale
    data = data.reshape(new_h, scale, new_w, scale).mean(axis=(1, 3))
    if to_int:
        data = np.uint8(data)
    return data


class SuperresDataset(Dataset):
    def __init__(self, dataset, scaling_factor=SCALE, features=FEATURE_LIST):
        super().__init__()
        self.hr_data = dataset
        self.scale = scaling_factor
        self.features = features
        self.transforms = transforms.Compose([
            transforms.Normalize(mean=means, std=[1.0]*len(means))
        ])
    def __len__(self):
        return len(self.hr_data['valid_time'].values)

    def __getitem__(self, index):
        hr_patches = []
        lr_patches = []

        for var in self.features:
              
            hr_patch = self.hr_data[var][index, :, :].values
            lr_patch = average_pooling(hr_patch, self.scale)

            lr_patches.append(torch.tensor(lr_patch, dtype=torch.float32))
            hr_patches.append(torch.tensor(hr_patch, dtype=torch.float32))
            

        hr_patches = torch.stack(hr_patches, axis=0)
        lr_patches = torch.stack(lr_patches, axis=0)
        hr_patches = self.transforms(hr_patches)
        lr_patches = self.transforms(lr_patches)
        # only return u10 and v10 of hr_patches
        
        return lr_patches, hr_patches[:2, :, :]

def initialize_dataset(config):
    data = load_dataset(DATA_PATH, config.train_start_date, config.test_end_date, PATCH_SIZE)
    train_data = data.sel(valid_time=slice(TRAIN_START, TRAIN_END))
    test_data = data.sel(valid_time=slice(TEST_START, TEST_END))

    train_data, train_scale = rescale_data(train_data)
    test_data, _ = rescale_data(test_data, custom_scale=train_scale)
    # print range of train and test data
    print("Train data range")
    print(train_scale)
    
    train_dataset = SuperresDataset(train_data)
    #save_means_stds(config.config_path, train_dataset, config.feature_list)
    test_dataset = SuperresDataset(test_data)
   
    return train_dataset, test_dataset

if __name__ == '__main__':
    train_set, test_set = initialize_dataset(config)
    print(len(train_set), len(test_set))
    hr, lr = train_set[0]
    hr2, lr2 = train_set[1]
    print(hr.shape, lr.shape)

