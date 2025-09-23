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
        x_binary = self.down_proj(x) > 0 # (B, 256, d), binary
        code = x_binary * 2 - 1 # (B, 256, d), -1 or 1
        code = code.to(x.dtype)
        x_vq = self.up_proj(code) # (B, 256, llm_hidden_size)

        return x_vq, code


from model.vq.vit import ViT
class LFQ_transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.down_proj = ViT(config)
        self.up_proj = nn.Sequential(
            nn.Linear(config.down_dim, config.llm_hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(config.llm_hidden_size, config.llm_hidden_size, bias=True)
        )

    def forward(self, x):
        """
        x: (B, 256, 4096)
        """
        x_binary = self.down_proj(x) > 0 # (B, 256, d), binary
        code = x_binary * 2 - 1 # (B, 256, d), -1 or 1
        code = code.to(x.dtype)
        x_vq = self.up_proj(code) # (B, 256, llm_hidden_size)

        return x_vq, code

class LFQ_autoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = ViT(config.encoder)
        self.decoder = ViT(config.decoder)
    
    def forward(self, x):
        """
        x: (B, 256, 4096)
        """
        x_binary = self.encoder(x) > 0 # (B, 256, 16)
        code = x_binary * 2 - 1 # (B, 256, 16), -1 or 1
        code = code.to(x.dtype)

        x_recon = self.decoder(code) # (B, 256, 4096)

        return x_recon, code


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load("config/vq/cos_rec_lfq.yaml")
    lfq = LFQ_autoencoder(config.model)
    x = torch.randn(1, 256, 4096)
    x_recon, code = lfq(x)
    print(code, x_recon.shape, code.shape)