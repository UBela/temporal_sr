from EDSR.train import CustomTrainer
from EDSR.model import EDSRModel
from super_image import TrainingArguments
from process_data import initialize_dataset
import torch
import time
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
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")



if config.pretraining:
    train_dataset, eval_dataset = initialize_dataset(config, test_year = config.test_year_pretraining)
    training_args =TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs)
    model = EDSRModel(in_channels=config.in_channels,
                    out_channels=config.out_channels,
                    feature_channels=config.feature_channels, 
                    scaling_factor=config.scaling_factor)

    model.to(device)
    
    trainer = CustomTrainer(model, training_args, train_dataset, eval_dataset)
    trainer.args.per_device_train_batch_size = config.train_batch_size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    print(f'Training arguments: {training_args}')
    train_start_time = time.time()
    trainer.train()
    train_end_time = time.time()
    print(f"Training time: {train_end_time - train_start_time} seconds")
    trainer.plot_metrics()
else:
    
    for year in range(config.test_years_start, config.test_years_end + 1):
        
        
        
        train_dataset, eval_dataset = initialize_dataset(config, test_year=year)
        training_args =TrainingArguments(
                output_dir=config.output_dir,
                num_train_epochs=config.num_train_epochs)
        
        
        pretrained_model = EDSRModel(in_channels=config.in_channels,
                            out_channels=config.out_channels,
                            feature_channels=config.feature_channels, 
                            scaling_factor=config.scaling_factor)

        #Inference
        print("Inference for year", year)
        
        pretrained_model.load_state_dict(torch.load(f"{config.model_path}/pytorch_model_{config.scaling_factor}x.pt", weights_only=False))
        pretrained_model.to(device)
        tester = CustomTrainer(pretrained_model, training_args, train_dataset, eval_dataset)
        tester.args.num_train_epochs = 1
        test_start_time = time.time()
        tester.eval(epoch=0, test_year=year)
        test_end_time = time.time()
        print(f"Inference time: {test_end_time - test_start_time} seconds.")