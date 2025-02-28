
import sys
sys.path.insert(0, '.')
sys.path.insert(1, '..')

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from box import Box
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from torcheval.metrics.functional import r2_score 
from DDIM.pipeline import CondDDIMPipeline
from utils import denormalize, calc_mae, calc_mse, AverageMeter

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

parser = argparse.ArgumentParser(description='Run model training with config')
parser.add_argument('config_path', type=str, help='Path to the config file')

args = parser.parse_args()
config = Box(load_config(args.config_path)['TrainingConfig'])

device = "cuda:0" if torch.cuda.is_available() else "cpu"

means = [config.mean_u10, config.mean_v10, config.mean_t2m, config.mean_d2m, config.mean_msl, config.mean_tp]
class DDIMTrainer():
    def __init__(self, train_dataset, test_dataset, model=None, optimizer=None, lr_scheduler=None, noise_scheduler=None, device=device):   
        self.train_losses = []
        self.eval_mses = []
        self.eval_maes = []
        self.eval_r2s = []
        self.eval_mses_samples = []
        self.eval_maes_samples = []
        self.eval_r2s_samples = []
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.noise_scheduler = noise_scheduler
        self.device = device
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError('train_dataset is not defined.')
        
        data_loader = DataLoader(
            self.train_dataset,
            batch_size = config.train_batch_size,
            shuffle=False,
        )
        return data_loader
    
    def get_eval_dataloader(self):
        eval_dataset = self.test_dataset if self.test_dataset else self.train_dataset
        data_loader = DataLoader(
            eval_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
        )
        return data_loader
        
    def _sine_encoding(self, noise):
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

    
    def train(self):
        # Initialize accelerator and tensorboard logging
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(config.output_dir, "logs"),
        )
        if accelerator.is_main_process:
            if config.output_dir is not None and not os.path.exists(config.output_dir):
                os.makedirs(config.output_dir, exist_ok=True)
            if config.model_path is not None and not os.path.exists(config.model_path):
                os.makedirs(config.model_path, exist_ok=True)

        train_dataloader = self.get_train_dataloader()
        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, train_dataloader, self.lr_scheduler
        )

        global_step = 0

        # Now you train the model
        for epoch in range(config.num_train_epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            self.model.train()
            epoch_loss = AverageMeter()
            for step, batch in enumerate(train_dataloader):
                low_res_images, clean_images = batch[0].to(self.device), batch[1].to(self.device)
                
                #clean_images = clean_images[:,int(config.sequence_length):int(2 * config.sequence_length),:,:]

                noise = torch.randn(clean_images.shape, device=clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                    dtype=torch.int64
                )

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
                
                noisy_images = noisy_images.to(dtype=torch.float16)
                
                with accelerator.accumulate(model):
                    
                    # add low res images as condition
                    #low res images are of shape (bs, seq_len * num_features, 32, 32)
                    # noisy images are of shape (bs, num_features, 32, 32)
                   
                    noisy_images = torch.cat([noisy_images, low_res_images], dim=1)
                    # Predict the noise residual
                    noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                    loss = F.mse_loss(noise_pred, noise)
                    epoch_loss.update(loss.item(), bs)
                    accelerator.backward(loss)

                    
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                print(f"Epoch: {epoch}, Loss: {loss.detach().item()}, LR: {lr_scheduler.get_last_lr()[0]}")
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1
                
            self.train_losses.append(epoch_loss.avg)
            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                pipeline = CondDDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=self.noise_scheduler)

                if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_train_epochs - 1:
                    mse, mae, r2, mse_samples, mae_samples, r2_samples = self.evaluate(epoch, pipeline, config.test_year_pretraining, device)
                    self.eval_mses.append(mse)
                    self.eval_maes.append(mae)
                    self.eval_r2s.append(r2)
                    self.eval_mses_samples.append(mse_samples)
                    self.eval_maes_samples.append(mae_samples)
                    self.eval_r2s_samples.append(r2_samples)
                
                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_train_epochs - 1:
                    print(f"Saving model at epoch {epoch}")
                    pipeline.save_pretrained(config.model_path)
                    
 
    def evaluate(self, epoch, pipeline,test_year, device): 
        eval_steps = 0
        total_mse = 0.0
        total_mae = 0.0
        eval_steps = 0
        total_mse_samples = 0.0
        total_mae_samples = 0.0
        output_preds = []
        output_preds_samples = []
        all_preds = []
        all_preds_samples = []
        all_labels = []
        test_dataloader = self.get_eval_dataloader()

        with torch.no_grad(): 
            for batch in test_dataloader: 

                lr_patches = batch[0].to(device)  

                #from hr only grab the middle timestep, ie features 3, 4, and 5
                #hr_patches = batch[1][:,int(config.sequence_length):int(config.sequence_length * 2),:,:].to(device)
                hr_patches = batch[1].to(device)    


                # prediction
                samples = pipeline(
                    batch_size=config.eval_batch_size,
                    image=lr_patches,
                    generator=torch.Generator(device=device),
                    num_images_per_cond=config.num_images_per_cond,
                    num_inference_steps=config.num_eval_timesteps,
                    output_type="np.array"
                ).images

                eval_steps += 1      
                if eval_steps % 10 == 0:
                    print(f"Completed {eval_steps} evaluation steps.")

                # reshape output samples --> shape (num_images_per_cond, batch_size, out_channels, H, W) 
                samples = torch.from_numpy(samples)
                bs = int(samples.shape[0]/config.num_images_per_cond)
                sample_preds = samples.view(config.num_images_per_cond, bs, config.out_channels, config.patch_size, config.patch_size)

                # loss computation
                lr_patches = lr_patches.view(-1, config.sequence_length, config.out_channels, config.patch_size, config.patch_size)
                lr_patches = lr_patches[:, int((config.sequence_length - 1) / 2), :, :,:].to(device)
                preds = sample_preds.mean(dim=0).to(device)

                loss = F.mse_loss(preds, hr_patches).to(device)
                print("Evaluation loss: {}".format(loss.item()))
                
                # plot samples
                if eval_steps <= 1:
                    
                    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
                    ax[0].imshow(sample_preds[0,0,0,:,:].cpu().numpy(), cmap='inferno')
                    ax[0].set_title(f'Single Sample Prediction 1')
                    ax[1].imshow(sample_preds[1,0,0,:,:].cpu().numpy(), cmap='inferno')
                    ax[1].set_title(f'Single Sample Prediction 2')
                    ax[2].imshow(lr_patches[0,0,:,:].cpu().numpy(), cmap='inferno')
                    ax[2].set_title(f'Low Res Input')
                    ax[3].imshow(hr_patches[0,0,:,:].cpu().numpy(), cmap='inferno')
                    ax[3].set_title(f'HR Ground Truth')
                    ax[4].imshow(preds[0,0,:,:].cpu().numpy(), cmap='inferno')
                    ax[4].set_title(f'Average Prediction')
                    plt.savefig(f'{config.output_dir}/output_epoch{epoch}.png')
                    plt.close()
                    for i in range(3):
                        print("Variable: {}".format(i))
                        print("HR max value: {}".format(hr_patches[:,i,:,:].max()))
                        print("HR min value: {}".format(hr_patches[:,i,:,:].min()))
                        print("HR mean value: {}".format(hr_patches[:,i,:,:].mean()))

                        print("Preds max value: {}".format(preds[:,i,:,:].max()))
                        print("Preds min value: {}".format(preds[:,i,:,:].min()))
                        print("Preds mean value: {}".format(preds[:,i,:,:].mean()))

                        print("LR max value: {}".format(lr_patches[:,i,:,:].max()))
                        print("LR min value: {}".format(lr_patches[:,i,:,:].min()))
                        print("LR mean value: {}".format(lr_patches[:,i,:,:].mean()))
                
                # collect preds
                preds = denormalize(preds, means)
                sample_preds = denormalize(sample_preds[0,:,:,:,:], means)
                hr_patches = denormalize(hr_patches, means)
                total_mse += calc_mse(preds.to('cpu'), hr_patches.to('cpu')).item()
                total_mae += calc_mae(preds.to('cpu'), hr_patches.to('cpu')).item()
                total_mse_samples += calc_mse(sample_preds.to('cpu'), hr_patches.to('cpu')).item()
                total_mae_samples += calc_mae(sample_preds.to('cpu'), hr_patches.to('cpu')).item()
                
                
                all_preds.append(preds.view(-1))
                all_preds_samples.append(sample_preds.view(-1))
                all_labels.append(hr_patches.view(-1))
                output_preds.append(preds)
                output_preds_samples.append(sample_preds)

            total_mse /= len(test_dataloader)
            total_mae /= len(test_dataloader)
            total_mse_samples /= len(test_dataloader)
            total_mae_samples /= len(test_dataloader)
            
            all_preds = torch.cat(all_preds)
            all_preds_samples = torch.cat(all_preds_samples)
            all_labels = torch.cat(all_labels)
            
            r2_score_val = r2_score(all_preds.to('cpu'), all_labels.to('cpu')).item()
            r2_score_samples = r2_score(all_preds_samples.to('cpu'), all_labels.to('cpu')).item()
            if epoch == config.num_train_epochs - 1:
                print("Final Evaluation")
            
            print(f'Epoch {epoch} - MSE: {total_mse}, MAE: {total_mae}, R2: {r2_score_val}')
            print(f'Epoch {epoch} - MSE Samples: {total_mse_samples}, MAE Samples: {total_mae_samples}, R2 Samples: {r2_score_samples}')
                
            if not config.pretraining:
                preds_tensor = torch.stack(output_preds)
                preds_samples_tensor = torch.stack(output_preds_samples)
                torch.save(preds_tensor, f'{config.preds_dir}/preds_DDIM_{test_year}_{config.scaling_factor}.pt')
                torch.save(preds_samples_tensor, f'{config.preds_dir}/preds_samples_DDIM_{test_year}_{config.scaling_factor}.pt')
                
                
        return total_mse, total_mae, r2_score_val, total_mse_samples, total_mae_samples, r2_score_samples
