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
BATCH_SIZE = 32
NUM_WORKERS = 2
IS_SHUFFLED = True

# wrapper function to load dataset
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
        
        
        
# get means for each variable for torch.transforms.Normalize
def get_means_stds(dataset):
    means = []
    stds = []
    for var in dataset:
        means.append(dataset[var].values.mean())
        stds.append(dataset[var].values.std())
    return means, stds

# Low resolution images, apply average pooling of scaleXscale
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
            os.makedirs(f'{hr_dir}/{var}')
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
        os.makedirs(f'{lr_dir}/{dataset[var]}')
        for i in range(len(dataset[var])):
            patch = dataset[var][:, :32, :32].values
            
            data = MinMaxScaler(feature_range=(0, 255)).fit(patch)
            data = np.uint8(data.transform(patch))
            data = average_pooling(data, scale, to_int=True)
            img = Image.fromarray(data)
            img.save(f'{lr_dir}/{var}/{i}.png')
    print('LR images done')
   
class SuperresDataset(Dataset):
    
    def __init__(self, dataset, scaling_factor=SCALE, features=FEATURE_LIST):
        super().__init__()
        self.hr_data = dataset
        self.scale = scaling_factor
        self.means, self.stds = get_means_stds(self.hr_data)
        
        self.features = features
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
      
        # zero mean and unit variance
       
        hr_patches = self.transform(hr_patches)
        lr_patches = self.transform(lr_patches)

        return hr_patches, lr_patches
    
def initialize_dataloader():
  
    data = load_dataset(DATA_PATH, START_DATE, END_DATE, PATCH_SIZE)
    # QUESTION NORMAILZE BOTH TRAIN AND TEST DATA WIITH SAME SCALE?
    #data, scale = normalize_data(data)
    
    train_data = data.sel(valid_time=slice(TRAIN_START, TRAIN_END))
    test_data = data.sel(valid_time=slice(TEST_START, TEST_END))
    train_data
   
    train_dataset = SuperresDataset(train_data)
    test_dataset = SuperresDataset(test_data)
    

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=IS_SHUFFLED)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=IS_SHUFFLED)
    return train_dataset, test_dataset, train_loader, test_loader

if __name__ == '__main__':
    
    train_set, test_set, train_dataloader, test_dataloader = initialize_dataloader()

    #get a random input tuple and turn all tensors inside into images and plot them

    hr, lr = train_set[0]
    print(hr.shape, lr.shape)
