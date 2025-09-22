import torch
import torch.nn as nn


class LFQ(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.down_proj = nn.Linear(config.input_dim, config.down_dim, bias=True)
        self.up_proj = nn.Sequential(
            nn.Linear(config.down_dim, config.llm_hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(config.llm_hidden_size, config.llm_hidden_size, bias=True)
        )

    def forward(self, x):
        """
        x: (B, 256, 4096)
        """
        x = self.down_proj(x) > 0 # (B, 256, d), binary
        code = x * 2 - 1 # (B, 256, d), -1 or 1
        x_vq = self.up_proj(code) # (B, 256, llm_hidden_size)

        return x_vq, code