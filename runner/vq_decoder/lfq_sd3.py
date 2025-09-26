import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import argparse
from omegaconf import OmegaConf
from einops import rearrange
from util.trainer import Trainer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

import logging
logging.basicConfig(level=logging.ERROR)

def pixel_shuffle(x, scale_factor=0.5):
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                int(c / (scale_factor * scale_factor)))

    x = x.permute(0, 2, 1, 3).contiguous()

    return x

def get_vit_feature(clip_encoder, pixel_values):
    vit_embeds = clip_encoder(
        pixel_values         = pixel_values,
        output_hidden_states = False,
        return_dict          = True
    ).last_hidden_state

    vit_embeds = vit_embeds[:, 1:, :]
    
    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
    vit_embeds = pixel_shuffle(vit_embeds, scale_factor=0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

    return vit_embeds

class MyTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def _load_models(self):
        from diffusers import AutoencoderKL
        from model.mmdit import load_mmdit_lfq
        from model.internvl.modeling_internvl_chat import InternVLChatModel

        vae = AutoencoderKL.from_pretrained(self.config.model.vae_path)
        clip_encoder = InternVLChatModel.from_pretrained(self.config.model.internvl_path).vision_model
        mmdit = load_mmdit_lfq(self.config.model.mmdit)

        vae.requires_grad_(False)
        clip_encoder.requires_grad_(False)

        self.vae = vae.to(self.device, self.dtype).eval()
        self.clip_encoder = clip_encoder.to(self.device, self.dtype).eval()
        self.model = mmdit

    def _load_dataloader(self):
        from util.dataloader import get_blip3o_dataloader
        self.dataloader = get_blip3o_dataloader(self.config.data, self.accelerator)

    def train(self):
        import copy
        from diffusers import FlowMatchEulerDiscreteScheduler
        from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.config.model.mmdit.sd3_5_path, subfolder="scheduler")
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)

        def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
            sigmas = noise_scheduler_copy.sigmas.to(device=self.device, dtype=dtype)
            schedule_timesteps = noise_scheduler_copy.timesteps.to(self.device)
            timesteps = timesteps.to(self.device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma

        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        training_done = False

        imagenet_mean = torch.tensor(IMAGENET_MEAN, device=self.accelerator.device, dtype=self.dtype).view(1, 3, 1, 1)
        imagenet_std = torch.tensor(IMAGENET_STD, device=self.accelerator.device, dtype=self.dtype).view(1, 3, 1, 1)

        while not training_done:
            for batch in self.dataloader:
                with self.accelerator.accumulate(self.model):
                    self.model.train()

                    pixel_values = batch["pixel_values"].to(self.device, self.dtype)
                    x_intern = (pixel_values - imagenet_mean) / imagenet_std
                    x_vae = pixel_values * 2 - 1

                    with torch.no_grad():
                        vit_feature = get_vit_feature(self.clip_encoder, x_intern) # (B, 256, 4096)
                        latents = self.vae.encode(x_vae).latent_dist.sample()
                        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                    
                    noise = torch.randn_like(latents)
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme = "logit_normal",
                        batch_size       = latents.shape[0],
                        logit_mean       = 0.0,
                        logit_std        = 1.0,
                        mode_scale       = 1.29,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=self.device)
                    sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=self.dtype)
                    noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

                    context = self.model.vit.forward_ste(vit_feature)

                    model_pred = self.model(
                        x           = noisy_model_input,
                        t           = timesteps,
                        context     = context,
                        y           = None,
                    )

                    model_pred = model_pred * (-sigmas) + noisy_model_input
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme="logit_normal", sigmas=sigmas)
                    target = latents

                    loss = torch.mean(
                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1,
                    ).mean()

                    self.optimizer.zero_grad()
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.params_to_learn, 1.0)
                        self.optimizer.step()

                        self.global_step += 1
                        self.progress_bar.update(1)

                        logs = dict(
                            loss = self.accelerator.gather(loss.detach()).mean().item(),
                        )
                        self.accelerator.log(logs, step=self.global_step)
                        self.progress_bar.set_postfix(**logs)

def main(args):
    config = OmegaConf.load(args.config)
    trainer = MyTrainer(config)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)