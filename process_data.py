import sys
sys.path.insert(0, '.')
sys.path.insert(1, '..')

import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import yaml
from box import Box
import argparse
from datetime import datetime
from utils import *
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

parser = argparse.ArgumentParser(description='Run model training with config')
parser.add_argument('config_path', type=str, help='Path to the config file')

args = parser.parse_args()
config = Box(load_config(args.config_path)['TrainingConfig'])

def load_dataset(data, start, end, patch_size, random_years=None):
    dataset = xr.open_dataset(data)
    dataset = dataset.isel(longitude=slice(0, patch_size), 
                            latitude=slice(0, patch_size))
    if random_years is not None:
        random_years = list(random_years)
        random_years.append(int(end.split('-')[0])) # add the test year to the list of random years
        dataset = dataset.sel(valid_time=dataset['valid_time'].dt.year.isin(random_years))
        data_list = []
        for year in random_years: 
            data_list.append(dataset.sel(valid_time = dataset['valid_time'].dt.year == year))
        dataset = xr.concat(data_list, dim='valid_time')
    else:
        
        dataset = dataset.sel(valid_time=slice(start, end))
    
    return dataset

def save_means_stds(config_path, dataset, feature_names):
    feature_tensors = []
    print("dataset length", len(dataset))
    for i in range(len(dataset)):
        
        _, hr_patches = dataset[i] 
        feature_tensors.append(hr_patches)
    
    features = torch.stack(feature_tensors, dim=0)
    means = features.view(features.shape[1], -1).mean(dim=1)  
    stds = features.view(features.shape[1], -1).std(dim=1) 
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file) or {}
    training_config = config.get("TrainingConfig", {})

    for mean, std, var in zip(means, stds, feature_names):
        training_config[f'mean_{var}'] = float(mean)
        training_config[f'std_{var}'] = float(std)
        print(f'{var}: mean={mean}, std={std}')
    config["TrainingConfig"] = training_config

    with open(config_path, 'w') as file:
        yaml.dump(config, file)

def downsampling(xarray, factor):
    return xarray.isel(latitude=slice(0, None, factor), longitude=slice(0, None, factor))

def get_random_years(config_path):
    random_years = list(map(int, np.random.choice(np.arange(1980, 2014), size=3, replace=False)))
    with open(config_path, 'r') as file:
        c = yaml.safe_load(file) or {}
    training_config = c.get("TrainingConfig", {})
    training_config['random_years'] = random_years
    c["TrainingConfig"] = training_config
    with open(config_path, 'w') as file:
        yaml.dump(c, file)  
    return random_years

class SuperresDataset(Dataset):
    def __init__(self, dataset, config, normalize = True):
        super().__init__()
        self.hr_data = dataset
        self.scale = config.scaling_factor
        self.features = config.feature_list[:3]
        self.normalize = normalize
        self.means = [config.mean_u10, config.mean_v10, config.mean_t2m]
        self.transforms = transforms.Compose([
            transforms.Normalize(mean=self.means, std=[1.0]*len(self.means))
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
        if self.normalize:
            
            hr_patches = self.transforms(hr_patches)
            lr_patches = self.transforms(lr_patches)
        
        return lr_patches, hr_patches
    
class SuperresDatasetDDIM(Dataset):
    """
    Create a dataset for super resolution using DDIM. Each item consists of 3 (t-1, t, t+1) LR patches and one HR patch at t. 
    The LR patches are preamtively upsampled to the HR patch size using bilinear interpolation.
    

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, dataset, config, normalize = True, seq_len_lr=3):
        super().__init__()
        self.hr_data = dataset
        self.scale = config.scaling_factor
        self.features = config.feature_list[:3]
        self.normalize = normalize
        self.means = [config.mean_u10, config.mean_v10, config.mean_t2m]
        self.transforms = transforms.Compose([
            transforms.Normalize(mean=self.means, std=[1.0]*len(self.means))
        ])
        self.seq_len_lr = config.sequence_length
        self.lr_data = average_pooling_xr(self.hr_data, self.scale)
       
        print(f"Dataset time steps: {len(self.hr_data['valid_time'].values)}")

    
    def __len__(self):
        return len(self.hr_data['valid_time'].values)
    
    def __getitem__(self, idx): 
        hr_patches = torch.stack([torch.tensor(self.hr_data[feature][idx, :, :].values, dtype=torch.float32) for feature in self.features])
        lr_patches = []
        for i in range(self.seq_len_lr):
            
            curr_index = idx - 1 + i
            curr_index = max(0, min(curr_index, len(self.hr_data['valid_time'].values) - 1))
            # turn lr_data into a numpy array first to speed up
            lr_patch = torch.tensor(np.array([self.lr_data[feature][curr_index, :, :].values for feature in self.features]), 
                                    dtype=torch.float32)

            if self.normalize: lr_patch = self.transforms(lr_patch)
            lr_patches.append(lr_patch)
            
        lr_patches = torch.cat(lr_patches, dim=0)

        lr_patches = F.interpolate(lr_patches.unsqueeze(0), 
                                    size=(hr_patches.shape[1], 
                                    hr_patches.shape[2]), 
                                    mode='bilinear', 
                                    align_corners=False).squeeze(0)
        
        if self.normalize: hr_patches = self.transforms(hr_patches)

        return lr_patches, hr_patches

        

def initialize_dataset(config, test_year):
    random_years = None
    test_start = f"{test_year}-01-01"
    test_end = f"{test_year}-12-31"
    
    if config.use_random_years:
        # For training randomly sample 3 years and save them in config for evaluation
        if config.pretraining:
            random_years = get_random_years(config.config_path)
        # For evaluation use the same 3 years as in pretraining to normalize the data
        else:
            random_years = config.random_years
        print(f"Random years: {random_years}")
    
    # in case train date is after test date, pass the oldest date as start and newest as end
    train_start = config.train_start_date
    train_end = config.train_end_date
    start = min(train_start, test_start)
    end = max(train_end, test_end)
   
    data = load_dataset(config.data_path, start, end, config.patch_size, random_years)
    if config.use_random_years:
        train_data = data.sel(valid_time=data['valid_time'].dt.year.isin(random_years))
    else:
        train_data = data.sel(valid_time=slice(config.train_start_date, config.train_end_date))
    data = data.sortby('valid_time')
    
    test_data = data.sel(valid_time=slice(test_start, test_end))
    train_data, train_scale = rescale_data(train_data)
   
    test_data, _ = rescale_data(test_data, custom_scale=train_scale)    
    
    if config.edsr:
        train_dataset = SuperresDataset(train_data, config, normalize=False)
        save_means_stds(config.config_path, train_dataset, config.feature_list)
        
        train_dataset = SuperresDataset(train_data, config)
        test_dataset = SuperresDataset(test_data, config)
    else:
        train_dataset = SuperresDatasetDDIM(train_data,config, normalize=False)
        save_means_stds(config.config_path, train_dataset, config.feature_list)
        train_dataset = SuperresDatasetDDIM(train_data, config)
        test_dataset = SuperresDatasetDDIM(test_data, config)
        
    return train_dataset, test_dataset
