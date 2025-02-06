import torch
from diffusers import UNet2DModel, DDIMPipeline
from box import Box
import argparse
import yaml

def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

parser = argparse.ArgumentParser(description='Run model training with configuration.')
parser.add_argument('config_path', type=str, help='Path to the configuration yaml file.')
args = parser.parse_args()
config = Box(load_config(args.config_path)['TrainingConfig'])

"""
From downscaling paper:
Our U-net utilize 4 downsampling blocks with increasing number of channels, with the number
channels being 64, 128, 256, 384. With the number of channels increasing as the spatial dimension
decrease. 
Lucas model had number of channels as (160, 320, 320, 640)
"""
def customize_model(config: Box):
    model = UNet2DModel(in_channels=config.in_channels,
                        out_channels=config.out_channels,
                        block_out_channels=(64, 128, 256, 384),
                        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
                        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
                        attention_head_dim=32,
                        )
    total_params = sum(p.numel() for p in model.parameters())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, total_params

def load_pretrained_model(config: Box):
    model = DDIMPipeline.from_pretrained(config.pretrained_model_path)
    model.config.sample_size = config.patch_size
    model.config.in_channels = config.in_channels 
    model.config.out_channels = config.out_channels
    
    model.config['sample_size'] = config.patch_size
    model.config['in_channels'] = config.in_channels
    model.config['out_channels'] = config.out_channels
    
    model.config['down_block_types'] = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D")
    model.config['up_block_types'] = ("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
    model.config['block_out_channels'] = (64, 128, 256, 384)
    model.config['attention_head_dim'] = 32
    out_channels = model.conv_in.out_channels
    model.conv_in = torch.nn.Conv2d(config.in_channels, config.out_channels, kernel_size=3, stride=1, padding=1)
    model.conv_out = torch.nn.Conv2d(out_channels, config.out_channels, kernel_size=3, stride=1, padding=1)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    total_params = sum(p.numel() for p in model.parameters())
    model.to(device)
    return model, total_params  