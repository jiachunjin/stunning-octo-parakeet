"""
Fix the internvl LM backbone, train the following:
1. the down projector
2. input MLP
3. diffusion head

with the following losses:
1. distillation loss between understanding with low dim clip feature
2. diffusion generation loss
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import copy
import torch
import argparse
from omegaconf import OmegaConf

from util.trainer import Trainer
from util.dataloader import get_llava_mix665k_dataloader
from model.internvl.modeling_internvl_chat import InternVLChatModel
from model.diff_mlp import SimpleMLPAdaLN


def equip_internvl(internvl, config):
    internvl.requires_grad_(False)
    down_proj = torch.nn.Linear(config.high_dim, config.down_dim)
    new_mlp1 = torch.nn.Linear(config.down_dim, 1024)
    diff_head = SimpleMLPAdaLN(
        in_channels    = config.diffhead.x_dim,
        model_channels = config.diffhead.hidden_size,
        out_channels   = config.diffhead.x_dim,
        z_channels     = config.diffhead.z_dim,
        num_res_blocks = config.diffhead.depth,
    )
    down_proj.requires_grad_(True)
    new_mlp1.requires_grad_(True)
    diff_head.requires_grad_(True)
    
    num_params = sum(p.numel() for p in down_proj.parameters())
    print(f"[down_proj] number of parameters: {num_params / 1e6} M")
    num_params = sum(p.numel() for p in new_mlp1.parameters())
    print(f"[new_mlp1] number of parameters: {num_params / 1e6} M")
    num_params = sum(p.numel() for p in diff_head.parameters())
    print(f"[diff_head] number of parameters: {num_params / 1e6} M")

    
    internvl.down_proj = down_proj
    internvl.new_mlp1 = new_mlp1
    internvl.diff_head = diff_head

    return internvl

class MyTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
    
    def _load_models(self):
        internvl = InternVLChatModel.from_pretrained(self.config.model.internvl_path)
        teacher = copy.deepcopy(internvl)
        teacher.requires_grad_(False)

        internvl = equip_internvl(internvl, self.config.model)
        if self.config.train.resume_path is not None:
            ckpt = torch.load(self.config.train.resume_path, map_location="cpu", weights_only=True)
            m, u = internvl.load_state_dict(ckpt, strict=False)
            print(f"missing keys: {m}, unmatched keys: {u}")

        self.teacher = teacher
        self.model = internvl

    def _load_dataloader(self):
        dataloader_und = get_llava_mix665k_dataloader()
        dataloader_gen = None

        self.dataloader_und = dataloader_und
    
    def train(self):
        print("Training...")
        exit(0)
        training_done = False

        while not training_done:
            for batch in self.dataloader_und:
                if batch is None:
                    continue
                with self.accelerator.accumulate(self.model):
                    ...

def main(args):
    config = OmegaConf.load(args.config)
    trainer = MyTrainer(config)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)