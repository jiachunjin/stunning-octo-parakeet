import torch
import torch.nn as nn
from model.vq.vit_basic import Block, FeedForward


class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.depth = config.depth
        self.num_heads = config.num_heads
        self.grid_size = config.grid_size

        self.precompute_pos = dict()
        self.input_proj = nn.Linear(config.input_dim, config.hidden_size)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.blocks = nn.ModuleList([Block(config.hidden_size, config.num_heads) for _ in range(config.depth)])
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.down_dim)

    def forward(self, x):
        """
        x: [B, L, D], original vit feature
        """
        pos = self.fetch_pos(self.grid_size, self.grid_size, x.device)
        B, L, D = x.shape

        x = self.input_proj(x)
        x = self.norm1(x).to(x.dtype)
        for block in self.blocks:
            x = block(x, pos)
        x = self.norm2(x)
        x = self.output_proj(x)

        return x