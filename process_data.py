import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
import os
from PIL import Image   
import json
import yaml
from box import Box
import argparse

#TODO put these in a config file
SCALE=2
DATA_PATH = 'data/data_stream-oper.nc'
START_DATE = '2020-01-01'
END_DATE = '2023-12-31'
PATCH_SIZE = 32
FEATURE_LIST = ['u10', 'v10', 'd2m', 't2m', 'msl', 'tp']
TRAIN_START = '2020-01-01'
TRAIN_END = '2021-01-01'
TEST_START = '2021-01-01'
TEST_END = '2022-01-01'


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

parser = argparse.ArgumentParser(description='Run model training with config')
parser.add_argument('config_path', type=str, help='Path to the config file')

args = parser.parse_args()
config = Box(load_config(args.config_path)['TrainingConfig'])



def load_dataset(data, start, end, patch_size):
    dataset = xr.open_dataset(data).sel(valid_time=slice(start, end)).isel(longitude=slice(0, patch_size), latitude=slice(0, patch_size))
    return dataset

def normalize_data(data):
    scale = {}
    for var in data:
        min_val = data[var].values.min()
        max_val = data[var].values.max()
        
        data[var] = (data[var] - min_val) / (max_val - min_val)
        scale[var] = {'min': min_val, 'max:': max_val}
        
        data[var] = 2 * data[var] - 1
        
        return data, scale
         
def get_means_stds(dataset, config_path):
    means = {}
    stds = {}

    for var in dataset:
        means[var] = float(dataset[var].values.mean())  
        stds[var] = float(dataset[var].values.std())    

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        if config is None:
            config = {}  
    config['means'] = means
    config['stds'] = stds

    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    return means, stds

def average_pooling(data, scale, to_int = False):
    new_h, new_w = data.shape[0]//scale, data.shape[1]//scale
    data = data.reshape(new_h,scale ,new_w,scale).mean(axis=(1, 3))
    if to_int:
        data = np.uint8(data)
    return data


def create_hr_images(dataset, hr_dir):
        if not os.path.exists(hr_dir):
            os.makedirs(hr_dir)

        for var in dataset:
            dir = f'{hr_dir}/{var}'
            if not os.path.exists(dir):
                os.makedirs(dir)
            for i in range(len(dataset[var])):
                patch = dataset[var][:, :32, :32].values
                
                data = MinMaxScaler(feature_range=(0, 255)).fit(patch)
                data = np.uint8(data.transform(patch))
                img = Image.fromarray(data)
                img.save(f'{hr_dir}/{var}/{i}.png')

        print('HR images done')

def create_lr_images(dataset, lr_dir, scale=SCALE):
    
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)

    for var in dataset:
        dir = f'{lr_dir}/{var}'
        if not os.path.exists(dir):
            os.makedirs(dir)
        for i in range(len(dataset[var])):
            patch = dataset[var][:, :32, :32].values
            
            data = MinMaxScaler(feature_range=(0, 255)).fit(patch)
            data = np.uint8(data.transform(patch))
            data = average_pooling(data, scale, to_int=True)
            img = Image.fromarray(data)
            img.save(f'{lr_dir}/{var}/{i}.png')
    print('LR images done')
   
class SuperresDataset(Dataset):
    
    def __init__(self, dataset, means, stds, scaling_factor=SCALE, features=FEATURE_LIST):
        super().__init__()
        self.hr_data = dataset
        self.scale = scaling_factor
        self.features = features
        self.means, self.stds = [means[var] for var in features], [stds[var] for var in features]
        self.transform = transforms.Compose([
            transforms.Normalize(mean=self.means, std=self.stds)
        ])
    def __len__(self):
        return len(self.hr_data['valid_time'].values)
    
    def __getitem__(self, index):
        hr_patches = []
        lr_patches = []
       
        for var in self.features:
            #print(var)
            hr_patch = self.hr_data[var][index, :, :].values
            lr_patch = average_pooling(hr_patch, self.scale)
           
            lr_patches.append(torch.tensor(lr_patch, dtype=torch.float32))
            hr_patches.append(torch.tensor(hr_patch, dtype=torch.float32))
            
        hr_patches = torch.stack(hr_patches, axis=0)
        lr_patches = torch.stack(lr_patches, axis=0)
      
        # zero mean and unit variance
       
        hr_patches = self.transform(hr_patches)
        lr_patches = self.transform(lr_patches)

        return lr_patches, hr_patches
    
def initialize_dataset(config):
  
    data = load_dataset(DATA_PATH, START_DATE, END_DATE, PATCH_SIZE)
    means, stds = get_means_stds(data, 'config.yaml')
    #data, scale = normalize_data(data)
    
    train_data = data.sel(valid_time=slice(TRAIN_START, TRAIN_END))
    test_data = data.sel(valid_time=slice(TEST_START, TEST_END))
    train_data
   
    train_dataset = SuperresDataset(train_data, means, stds)
    test_dataset = SuperresDataset(test_data, means, stds)
    
    return train_dataset, test_dataset

if __name__ == '__main__':
    
    train_set, test_set = initialize_dataset(config)

    hr, lr = train_set[0]
    print(hr.shape, lr.shape)


#TODO get mean and var after avg pooling, write into yaml