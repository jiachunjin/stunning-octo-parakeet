import torch
from .mmditx import MMDiTX

import os
from safetensors.torch import load_file

def load_mmdit_lfq(config):
    device = torch.device("cpu")
    dtype = torch.bfloat16

    # --------- stable diffusion 3.5 default parameters ---------
    patch_size = 2
    depth = 24
    pos_embed_max_size = 384
    num_patches = 147456
    adm_in_channels = 2048
    qk_norm = "rms"
    x_block_self_attn_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    context_embedder_config = {
        "target": "torch.nn.Linear",
        "params": {
            "in_features": config.lfq_vit.output_dim,
            "out_features": 1536,
        },
    }

    transformer = MMDiTX(
        input_size               = None,
        pos_embed_scaling_factor = None,
        pos_embed_offset         = None,
        pos_embed_max_size       = pos_embed_max_size,
        patch_size               = patch_size,
        in_channels              = 16,
        depth                    = depth,
        num_patches              = num_patches,
        adm_in_channels          = adm_in_channels,
        context_embedder_config  = context_embedder_config,
        qk_norm                  = qk_norm,
        x_block_self_attn_layers = x_block_self_attn_layers,
        device                   = device,
        dtype                    = dtype,
        verbose                  = False,
    )
    # --------- add lfq vit ---------
    from model.vq.vit import ViT
    vit = ViT(config.lfq_vit)
    num_params = sum(p.numel() for p in vit.parameters())
    print(f"vit has {num_params / 1e6} M parameters")
    vit.requires_grad_(True)
    transformer.vit = vit
    # --------- load pretrained weights ---------
    if config.load_pretrained:
        ckpt = load_file(os.path.join(config.sd3_5_path, "sd3.5_medium.safetensors"))
        new_ckpt = {}
        prefix = "model.diffusion_model."
        for k, v in ckpt.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                new_ckpt[new_key] = v
        del new_ckpt["context_embedder.weight"]
        m, u = transformer.load_state_dict(new_ckpt, strict=False)
        print(f"missing keys: {m}")
        print(f"unexpected keys: {u}")

    # --------- define trainable parameters ---------
    transformer.requires_grad_(False)
    transformer.context_embedder.requires_grad_(True)
    num_para = sum(p.numel() for p in transformer.context_embedder.parameters())
    print("context_embedder parameters: ", num_para / 1e6)
    for name, param in transformer.named_parameters():
        if "context_block" in name:
            param.requires_grad_(True)

    num_para = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("total parameters: ", num_para / 1e6)

    return transformer