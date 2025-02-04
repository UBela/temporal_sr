from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
import torch
from torch.utils.data import Dataloader
import torch.nn.functional as F
from box import Box
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from torcheval.metrics.functional import r2_score 
from pipeline import CondDDIMPipeline


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

parser = argparse.ArgumentParser(description='Run model training with config')
parser.add_argument('config_path', type=str, help='Path to the config file')

args = parser.parse_args()
config = Box(load_config(args.config_path)['TrainingConfig'])

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Trainer():
    def __init__(self, model, args, train_dataset, test_dataset):
        self.train_losses = []
        self.eval_mses = []
        self.eval_maes = []
        self.eval_r2s = []
        self.train_dataset = train_dataset
        self. test_dataset = test_dataset
    
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError('train_dataset is not defined.')
        
        data_loader = Dataloader(
            self.train_dataset,
            batch_size = config.train_batch_size,
            shuffle=False,
        )
    
    def get_eval_dataloader(self):
        eval_dataset = self.eval_dataset if self.eval_dataset else self.train_dataset
        data_loader = DataLoader(
            eval_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
        )
        
    def sine_encoding(self, noise):
        """Create sinusoidal encoding of noise by splitting it into a set of frequencies.
        Args:
            noise (_type_): _description_
        """
        embedding_min_freq = config.min_freq # 1.0
        embedding_max_freq = config.max_freq # 10000.0
        
        frequencies = torch.exp(torch.linspace(
                            start=torch.log(embedding_min_freq),
                            end=torch.log(embedding_max_freq),
                            steps=noise.shape[-1],
                            device=noise.device,))
        angular_speeds = 2 * np.pi * frequencies
        embedding = torch.cat([torch.sin(angular_speeds * noise), torch.cos(angular_speeds * noise)], dim=-1)
    return embedding

    def plot_metrics(self):
        epochs = range(len(self.train_losses))

        train_losses_log = torch.log(torch.tensor(self.train_losses))
        eval_mses_log = torch.log(torch.tensor(self.eval_mses))
        eval_maes_log = torch.log(torch.tensor(self.eval_maes))

        plt.figure(figsize=(18, 5))

        # Plot training loss
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_losses_log, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Log MSE Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()

        # Plot evaluation MSE
        plt.subplot(1, 3, 2)
        plt.plot(epochs, eval_mses_log, label="Evaluation MSE", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Log MSE")
        plt.title("Evaluation MSE Over Epochs")
        plt.legend()

        # Plot evaluation MAE
        plt.subplot(1, 3, 3)
        plt.plot(epochs, eval_maes_log, label="Evaluation MAE", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Log MAE")
        plt.title("Evaluation MAE Over Epochs")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{config.output_dir}/training_eval_metrics_{config.scaling_factor}x.png')
        plt.close()

    def train():
        pass
    
    def evaluate():
        pass
    
    
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
                    
                    
def evaluate(config, epoch, pipeline, test_dataloader):
    mses = []
    maes = []
    r2_scores = []
    eval_steps = 0
    # Evaluate the model
    pipeline.eval()
    with torch.no_grad():
        for i, (low_res_images, high_res_images) in enumerate(test_dataloader):
            
            low_res_images, high_res_images = low_res_images.to(device), high_res_images.to(device)
            
            samples = pipeline(
                batch_size = config.eval_batch_size,
                image = low_res_images,
                generator = torch.Generator,
                num_images_per_cond = config.num_images_per_cond,
                num_inference_steps = config.num_inference_steps,
                output_type = 'np.array'
            ).images
            samples = torch.from_numpy(samples)
            
            eval_steps += 1
            
            
            batch_size = int(samples.shape[0] / config.num_images_per_cond)
            sample_preds = samples.view(config.num_images_per_cond,
                                        batch_size,
                                        config.out_channels,
                                        config.patch_size,
                                        config.patch_size)
            
            preds = sample_preds.mean(dim=0)
            
            if eval_steps <= 1:
                fig, ax = plt.subplots(1, 5, figsize=(25, 5))
                ax[0].imshow(sample_preds[0,0,0,:,:].cpu().numpy(), cmap='inferno')
                ax[0].set_title(f'Single Sample Prediction 1')
                ax[1]imshow(sample_preds[1,0,0,:,:].cpu().numpy(), cmap='inferno')
                ax[1]set_title(f'Single Sample Prediction 2')
                ax[2]imshow(low_res_image[0,0,0,:,:].cpu().numpy(), cmap='inferno')
                ax[2]set_title(f'Low Res Input')
                ax[3].imshow(high_res_images[0,0,:,:].cpu().numpy(), cmap='inferno')
                ax[3].set_title(f'HR Ground Truth')
                ax[4].imshow(preds[0,0,:,:].cpu().numpy(), cmap='inferno')
                ax[4].set_title(f'Average Prediction')
                plt.savefig(f'{config.output_dir}/output_epoch{epoch}.png')
                plt.close()
            
            
            
            
            
           