from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
import torch
import torch.nn.functional as F
from box import Box
import argparse
import yaml
import numpy as np
from pipeline import CondDDIMPipeline
from ..process_data import initialize_dataset
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

parser = argparse.ArgumentParser(description='Run model training with config')
parser.add_argument('config_path', type=str, help='Path to the config file')

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def sine_encoding(noise):
    """Create sinusoidal encoding of noise by splitting it into a set of frequencies.
    Args:
        noise (_type_): _description_
    """
    embedding_min_freq = 1.0
    embedding_max_freq = 10000.0
    
    frequencies = torch.exp(torch.linspace(
                        start=torch.log(embedding_min_freq),
                        end=torch.log(embedding_max_freq),
                        steps=noise.shape[-1],
                        device=noise.device,))
    angular_speeds = 2 * np.pi * frequencies
    embedding = torch.cat([torch.sin(angular_speeds * noise), torch.cos(angular_speeds * noise)], dim=-1)
    return embedding
    

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            
            clean_images, low_res_images = batch.to(device)
            
            
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            noisy_images = noisy_images.to(dtype=torch.float16)
            
            

            with accelerator.accumulate(model):
                # Predict the noise residual
                # add low res images as condition
                #low res images are of shape (bs, 3, 3, 32, 32)
                # nosiy images are of shape (bs, 3, 32, 32)
                
                # -> concat to (bs, 4, 3, 32, 32)
                
                noisy_images = torch.cat([noisy_images, low_res_images], dim=1)
                
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = CondDDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)
                    
                    
def evaluate(config, epoch, pipeline):
    # Load the validation dataset
    val_dataset = initialize_dataset(config, config.test_year)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    # Evaluate the model
    pipeline.eval()
    with torch.no_grad():
        for i, (low_res_images, high_res_images) in enumerate(val_dataloader):
            low_res_images, high_res_images = low_res_images.to(device), high_res_images.to(device)
            noisy_images = pipeline(low_res_images, return_dict=False)[0]
            loss = F.mse_loss(noisy_images, high_res_images)
            if i == 0:
                pipeline.save_image(noisy_images, high_res_images, epoch, config.output_dir)
            logs = {"val_loss": loss.item()}
            accelerator.log(logs, step=epoch)
    pipeline.train()