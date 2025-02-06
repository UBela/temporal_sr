from DDIM.train import DDIMTrainer
from DDIM.model import load_config, customize_model, load_pretrained_model
from process_data import initialize_dataset
from diffusers import DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from DDIM.pipeline import CondDDIMPipeline
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
if config.pretraining:
    train_dataset, eval_dataset = initialize_dataset(config, test_year = config.test_year_pretraining)
    
    model, total_params = customize_model(config)
    print(f"Total number of parameters: {total_params}")
    print(f"Device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.lr_warmup_steps, 
        num_training_steps= (len(train_dataset) // config.train_batch_size) * config.num_train_epochs)
    
    noise_scheduler = DDIMScheduler(num_train_timesteps=config.num_train_timesteps)
    
    
    trainer = DDIMTrainer(
        train_dataset=train_dataset, 
        test_dataset=eval_dataset, 
        model=model, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler, 
        noise_scheduler=noise_scheduler, 
        device=device)
    train_start_time = time.time()
    trainer.train()
    train_end_time = time.time()
    trainer.plot_metrics()
    print(f"Training time: {train_end_time - train_start_time} seconds")
else:
    
    # Inference
    pretrained_pipeline = CondDDIMPipeline.from_pretrained(config.model_path, use_safetensors=True).to(device)
    
    for year in range(config.test_years_start, config.test_years_end + 1):
        print(f"Inference for year {year}")

        train_dataset, eval_dataset = initialize_dataset(config, test_year=year)
        
        tester = DDIMTrainer(train_dataset=None, test_dataset=eval_dataset, device=device)
    
        eval_start_time = time.time()

        tester.evaluate(epoch=0, pipeline=pretrained_pipeline, test_year=year)

        eval_end_time = time.time()
        print(f"Evaluation time: {eval_end_time - eval_start_time} seconds")
        print("Inference done for year", year)
