import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from model.vq.vit import ViT
from mmdit.mmdit_generalized_pytorch import MMDiT


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LFQDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vit = ViT(config.vit)
        self.mmdit = MMDiT(
            depth = config.mmdit.depth,
            dim_modalities = (config.bottleneck_dim, config.mmdit.vae_channel),
            dim_cond = config.mmdit.hidden_size,
            qk_rmsnorm = True
        )
        self.t_embedder = TimestepEmbedder(config.mmdit.hidden_size)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, 56 * 56, config.hidden_size), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.vit.pos_embed.shape[-1], int(self.config.vit.grid_size ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        

    def forward(self, vit_features, latents, t):
        """
        vit_features: (B, 256, 4096)
        latents: (B, C, H, W)
        """
        latents = rearrange(latents, "b c h w -> b (h w) c")
        features_down = self.vit(vit_features) + self.pos_embed
        p = torch.sigmoid(features_down)
        p_ = (p > 0.5).to(vit_features.dtype)
        feature_bin = p + (p_ - p).detach() # (B, 256, 16)

        conditions, latents = self.mmdit(
            modality_tokens = (feature_bin, latents),
            time_cond = self.t_embedder(t, vit_features.dtype)
        )

        return conditions, latents


if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("/Users/orres/Playground/stunning-octo-parakeet/config/vq_decoder/mmdit_decoder.yaml")
    decoder = LFQDecoder(config.model.lfq_decoder)
    B = 2
    vit_features = torch.randn(B, 256, 4096)
    latents = torch.randn(B, 16, 28, 28)
    t = torch.randint(0, 1000, (B,))
    conditions, latents = decoder(vit_features, latents, t)
    print(conditions.shape, latents.shape)