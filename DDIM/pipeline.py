
from typing import List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline, DDIMScheduler, ImagePipelineOutput, UNet2DModel
import inspect
class CondDDIMPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()

        # make sure scheduler can always be converted to DDIM
        scheduler = DDIMScheduler.from_config(scheduler.config)

        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        image: torch.Tensor = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_images_per_condition: Optional[int] = 1,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:

        batch_size, c, height, width = image.shape
        
        generator = generator or torch.Generator(device=self._execution_device)
        
        latents_shape = (batch_size * num_images_per_condition, self.unet.config.out_channels, height, width)
        
        latents = torch.randn(latents_shape, device=self._execution_device, generator=generator)
        latents_dtype = next(self.unet.parameters()).dtype
        
        image = torch.cat([image] * num_images_per_condition, dim=0).to(latents_dtype)
        image = image.to(self._execution_device, dtype=latents_dtype)
    
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        
        latents *= self.scheduler.init_noise_sigma
        
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_kwargs = {}   
        if accepts_eta:
            extra_kwargs['eta'] = eta
        for t in self.progress_bar(self.scheduler.timesteps):
            
            latents_input = torch.cat([latents, image], dim=1)
            latents_input = self.scheduler.scale_model_input(latents_input, t)
            # 1. predict noise model_output
            model_output = self.unet(latents_input, t).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            latents = self.scheduler.step(
                model_output, t, latents, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample

          

        image = latents.cpu().numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)