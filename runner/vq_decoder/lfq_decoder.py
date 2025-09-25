import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import argparse
from omegaconf import OmegaConf
from util.trainer import Trainer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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
        from model.decoder.lfq_decoder import LFQDecoder
        from model.internvl.modeling_internvl_chat import InternVLChatModel

        vae = AutoencoderKL.from_pretrained(self.config.model.vae_path)
        clip_encoder = InternVLChatModel.from_pretrained(self.config.model.internvl_path).vision_model
        lfq_decoder = LFQDecoder(self.config.model.lfq_decoder)

        vae.requires_grad_(False)
        clip_encoder.requires_grad_(False)

        self.vae = vae.to(self.device, self.dtype).eval()
        self.clip_encoder = clip_encoder.to(self.device, self.dtype).eval()
        self.model = lfq_decoder

    def _load_dataloader(self):
        from util.dataloader import get_blip3o_dataloader
        self.dataloader = get_blip3o_dataloader(self.config.data, self.accelerator)

    def train(self):
        from diffusers import DDPMScheduler

        train_scheduler = DDPMScheduler(
            beta_schedule          = "scaled_linear",
            beta_start             = 0.00085,
            beta_end               = 0.012,
            num_train_timesteps    = 1000,
            clip_sample            = False,
            prediction_type        = "v_prediction",
            steps_offset           = 1,
            trained_betas          = None,
            timestep_spacing       = "trailing",
            rescale_betas_zero_snr = True
        )

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
                        vit_feature = get_vit_feature(self.clip_encoder, x_intern)
                        latents = self.vae.encode(x_vae).latent_dist.sample()
                    
                    self.accelerator.print(vit_feature.shape, latents.shape)
                    exit(0)


def main(args):
    config = OmegaConf.load(args.config)
    trainer = MyTrainer(config)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)