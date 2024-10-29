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

def load_dataset(data, start, end, patch_size):
    dataset = xr.open_dataset(data).sel(valid_time=slice(start, end)).isel(
        longitude=slice(0, patch_size), latitude=slice(0, patch_size))
    return dataset

def normalize_data(data):
    scale = {}
    for var in data:
        min_val = data[var].values.min()
        max_val = data[var].values.max()
        
        data[var] = (data[var] - min_val) / (max_val - min_val)
        scale[var] = {'min': min_val, 'max': max_val}
        data[var] = 2 * data[var] - 1
        
    return data, scale

def get_means_stds(dataset, config_path):
    means = {}
    stds = {}

    for var in dataset:
        means[var] = float(dataset[var].values.mean())
        stds[var] = float(dataset[var].values.std())

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file) or {}

    config['means'] = means
    config['stds'] = stds

    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    return means, stds

def average_pooling(data, scale, to_int=False):
    new_h, new_w = data.shape[0] // scale, data.shape[1] // scale
    data = data.reshape(new_h, scale, new_w, scale).mean(axis=(1, 3))
    if to_int:
        data = np.uint8(data)
    return data

class SuperresDataset(Dataset):
    def __init__(self, dataset, means, stds, scaling_factor=SCALE, features=FEATURE_LIST):
        super().__init__()
        self.hr_data = dataset
        self.scale = scaling_factor
        self.features = features
        self.means = [means[var] for var in features]
        self.stds = [stds[var] for var in features]
        self.transform = transforms.Compose([
            transforms.Normalize(mean=self.means, std=self.stds)
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

        hr_patches = self.transform(hr_patches)
        lr_patches = self.transform(lr_patches)
        return lr_patches, hr_patches[:2, :, :]

def initialize_dataset(config):
    data = load_dataset(DATA_PATH, config.train_start_date, config.test_end_date, PATCH_SIZE)
    means, stds = get_means_stds(data, config.config_path)

    train_data = data.sel(valid_time=slice(TRAIN_START, TRAIN_END))
    test_data = data.sel(valid_time=slice(TEST_START, TEST_END))

    train_dataset = SuperresDataset(train_data, means, stds)
    test_dataset = SuperresDataset(test_data, means, stds)

    return train_dataset, test_dataset

if __name__ == '__main__':
    train_set, test_set = initialize_dataset(config)
    print(len(train_set), len(test_set))
    hr, lr = train_set[0]
    print(hr.shape, lr.shape)


#TODO model output weird, check if normalization is correct