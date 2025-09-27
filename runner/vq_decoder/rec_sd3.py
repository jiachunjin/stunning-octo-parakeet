import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
from tqdm import tqdm

@torch.no_grad()
def sample_sd3_5(
    transformer,
    vae,
    noise_scheduler,
    device,
    dtype, 
    context,
    batch_size          = 1,
    height              = 192,
    width               = 192,
    num_inference_steps = 20,
    guidance_scale      = 1.0,
    seed                = None,
    # multi_modal_context = False,
):
    if seed is not None:
        torch.manual_seed(seed)
    
    transformer.eval()
    
    latent_height = height // 8
    latent_width = width // 8
    
    latents = torch.randn(
        (batch_size, 16, latent_height, latent_width),
        device = device,
        dtype  = dtype
    )

    noise_scheduler.set_timesteps(num_inference_steps)
    timesteps = noise_scheduler.timesteps.to(device=device, dtype=dtype)
    
    for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
        if t.ndim == 0:
            t = t.unsqueeze(0)
        t = t.repeat(batch_size)

        latent_model_input = latents

        if guidance_scale > 1.0:
            latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)
            t = torch.cat([t, t], dim=0)
            context_ = torch.cat([context, torch.zeros_like(context, device=context.device, dtype=context.dtype)], dim=0)
        else:
            context_ = context

        noise_pred = transformer(
            x           = latent_model_input,
            t           = t,
            context     = context_,
            y           = None,
            # multi_modal_context = multi_modal_context,
        )

        if guidance_scale > 1.0:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        step_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=t[0] if t.ndim > 0 else t,
            sample=latents,
            return_dict=False,
        )
        latents = step_output[0]
    
    latents = 1 / vae.config.scaling_factor * latents + vae.config.shift_factor
    image = vae.decode(latents).sample
    
    image = (image / 2 + 0.5).clamp(0, 1)
    
    return image

@torch.no_grad()
def recon_lfq_sd3():
    # ---------- load model ----------
    import os
    from PIL import Image
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from model.mmdit import load_mmdit_lfq
    from model.internvl.modeling_internvl_chat import InternVLChatModel

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    exp_dir = "/data/phd/jinjiachun/experiment/vq_decoder/0926_lfq_sd3_decoder_fulltune"
    step = 56000

    config_path = os.path.join(exp_dir, "config.yaml")
    config = OmegaConf.load(config_path)
    ckpt_path = os.path.join(exp_dir, f"lfq_sd3_decoder-{config.train.exp_name}-{step}")

    vae = AutoencoderKL.from_pretrained(config.model.vae_path)
    clip_encoder = InternVLChatModel.from_pretrained(config.model.internvl_path).vision_model
    mmdit = load_mmdit_lfq(config.model.mmdit)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.model.mmdit.sd3_5_path, subfolder="scheduler")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    mmdit.load_state_dict(ckpt, strict=True)
    mmdit = mmdit.to(device, dtype).eval()
    vae = vae.to(device, dtype).eval()
    clip_encoder = clip_encoder.to(device, dtype).eval()

    # ---------- load images ----------
    import torchvision.transforms as pth_transforms
    from runner.vq_decoder.lfq_decoder import get_vit_feature

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    images = [
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/letter.jpeg").convert("RGB"),
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/letter1.webp").convert("RGB"),
        Image.open("/data/phd/jinjiachun/codebase/connector/asset/kobe.png").convert("RGB"),
        Image.open("/data/phd/jinjiachun/codebase/connector/asset/004.jpg").convert("RGB"),
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/einstein.jpg"),
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/jobs.jpg"),
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/mcdonald.jpg"),
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/messi_1.jpg"),
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/messi_2.webp"),
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/messi.jpg"),
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/ronaldo.jpg"),
        Image.open("/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/trump.jpg"),
    ]

    vae_transform = pth_transforms.Compose([
        pth_transforms.Resize(448, max_size=None),
        pth_transforms.CenterCrop(448),
        pth_transforms.ToTensor(),
    ])
    x_list = []
    for img in images:
        x_list.append(vae_transform(img).unsqueeze(0).to(device, dtype))
    x = torch.cat(x_list, dim=0)

    imagenet_mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    x = (x - imagenet_mean) / imagenet_std

    
    
    vit_feature = get_vit_feature(clip_encoder, x) # (B, 256, 4096)
    context = mmdit.vit.get_code(vit_feature)
    print(context)

    samples = sample_sd3_5(
        transformer         = mmdit,
        vae                 = vae,
        noise_scheduler     = noise_scheduler,
        device              = device,
        dtype               = dtype,
        context             = context,
        batch_size          = context.shape[0],
        height              = 448,
        width               = 448,
        num_inference_steps = 20,
        guidance_scale      = 1.0,
        seed                = 42
    )

    print(samples.shape)

    import torchvision.utils as vutils
    sample_path = f"assets/lfq_decoder/{config.train.exp_name}_{step}.png"
    vutils.save_image(samples, sample_path, nrow=2, normalize=False)
    print(f"Samples saved to {sample_path}")    


if __name__ == "__main__":
    recon_lfq_sd3()