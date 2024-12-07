import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from torchvision import transforms
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

DATA_PATH = config.data_path
SCALE = config.scaling_factor
PATCH_SIZE = config.patch_size
FEATURE_LIST = config.feature_list
TRAIN_START = config.train_start_date
TRAIN_END = config.train_end_date
n_in_features = config.in_channels
means = [config.mean_u10, config.mean_v10, config.mean_t2m, config.mean_d2m, config.mean_msl, config.mean_tp]
FEATURE_LIST = FEATURE_LIST[:n_in_features]
means = means[:n_in_features]


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
        dataset = xr.open_dataset(data).sel(valid_time=slice(start, end))
    
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
    feature_tensors = []
    for i in range(len(dataset)):
        lr_patches, _ = dataset[i]  
        feature_tensors.append(lr_patches)
    
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


def average_pooling(data, scale, to_int=False):
    new_h, new_w = data.shape[0] // scale, data.shape[1] // scale
    data = data.reshape(new_h, scale, new_w, scale).mean(axis=(1, 3))
    if to_int:
        data = np.uint8(data)
    return data

def get_random_years(config_path):
    random_years = np.random.choice(np.arange(1980, 2014), size=3, replace=False)
            
    with open(config_path, 'r') as file:
        c = yaml.safe_load(file) or {}
    training_config = c.get("TrainingConfig", {})
    training_config['random_years'] = random_years
    c["TrainingConfig"] = training_config
    with open(config_path, 'w') as file:
        yaml.dump(c, file)  
    return random_years

class SuperresDataset(Dataset):
    def __init__(self, dataset, scaling_factor=SCALE, features=FEATURE_LIST, normalize = True):
        super().__init__()
        self.hr_data = dataset# Use config values instead of hardcoding

        self.scale = scaling_factor
        self.features = features
        self.normalize = normalize
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
        if self.normalize:
            
            hr_patches = self.transforms(hr_patches)
            lr_patches = self.transforms(lr_patches)
        
        return lr_patches, hr_patches

def initialize_dataset(config, test_year):
    random_years = None
    TEST_START = f"{test_year}-01-01"
    TEST_END = f"{test_year}-12-31"
    
    if config.use_random_years:
        # For training randomly sample 3 years and save them in config for evaluation
        if config.pretraining:
            random_years = get_random_years(config.config_path)
        # For evaluation use the same 3 years as in pretraining to normalize the data
        else:
            random_years = config.random_years
        print(f"Random years: {random_years}")
        
    data = load_dataset(DATA_PATH, TRAIN_START, TEST_END, PATCH_SIZE, random_years=random_years)
    
    if config.use_random_years:
        train_data = data.sel(valid_time=data['valid_time'].dt.year.isin(random_years))
    else:
        train_data = data.sel(valid_time=slice(TRAIN_START, TRAIN_END))
        
    test_data = data.sel(valid_time=slice(TEST_START, TEST_END))

    train_data, train_scale = rescale_data(train_data)
    test_data, _ = rescale_data(test_data, custom_scale=train_scale)    
    
    train_dataset = SuperresDataset(train_data, normalize=False)
    save_means_stds(config.config_path, train_dataset, config.feature_list)
    
    train_dataset = SuperresDataset(train_data)
    test_dataset = SuperresDataset(test_data)
   
    return train_dataset, test_dataset


