import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch


@torch.no_grad()
def recon_lfq():
    # ---------- load model ----------
    import os
    from PIL import Image
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL
    from model.decoder.lfq_decoder import LFQDecoder
    from model.internvl.modeling_internvl_chat import InternVLChatModel

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    exp_dir = "/data/phd/jinjiachun/experiment/vq_decoder/0925_lfq_decoder_add_head_tail"
    step = 22000
    config_path = os.path.join(exp_dir, "config.yaml")
    config = OmegaConf.load(config_path)
    ckpt_path = os.path.join(exp_dir, f"lfq_decoder-{config.train.exp_name}-{step}")

    vae = AutoencoderKL.from_pretrained(config.model.vae_path)
    clip_encoder = InternVLChatModel.from_pretrained(config.model.internvl_path).vision_model
    lfq_decoder = LFQDecoder(config.model.lfq_decoder)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    lfq_decoder.load_state_dict(ckpt, strict=True)
    lfq_decoder = lfq_decoder.to(device, dtype).eval()
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

    vit_feature = get_vit_feature(clip_encoder, x)

    # ---------- reconstruct ----------
    from diffusers import DDIMScheduler
    from tqdm import tqdm
    import numpy as np

    scheduler = DDIMScheduler(
        beta_schedule          = "scaled_linear",
        beta_start             = 0.00085,
        beta_end               = 0.012,
        num_train_timesteps    = 1000,
        clip_sample            = False,
        prediction_type        = "v_prediction",
        set_alpha_to_one       = True,
        steps_offset           = 1,
        trained_betas          = None,
        timestep_spacing       = "trailing",
        rescale_betas_zero_snr = True
    )
    scheduler.set_timesteps(50)
    B = 16
    x = torch.randn((B, 16, 56, 56), device=device, dtype=dtype)
    x *= scheduler.init_noise_sigma

    for t in tqdm(scheduler.timesteps):
        x = scheduler.scale_model_input(x, t)
        t_sample = torch.as_tensor([t], device=device)
        _, noise_pred = lfq_decoder(vit_feature, x, t_sample)
        x = scheduler.step(noise_pred, t, x).prev_sample    

    x = vae.decode(x)
    x = (x + 1) / 2
    x = x.clamp(0, 1)
    x = x.permute(0, 2, 3, 1)
    x = x.cpu().numpy()
    x = (x * 255).astype(np.uint8)
    x = Image.fromarray(x)
    x.save("reconstruction.png")

    print("reconstruction done")


if __name__ == "__main__":
    recon_lfq()