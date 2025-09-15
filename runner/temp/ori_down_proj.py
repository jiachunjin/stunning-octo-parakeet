import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from model.internvl.modeling_internvl_chat import InternVLChatModel
from runner.down_proj.llava_distill import add_down_proj


@torch.inference_mode()
def test_ori_down_proj():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    config = OmegaConf.load("config/down_proj/llava_distill.yaml")
    internvl_path = "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3_5-1B"
    model_name = internvl_path.split("/")[-1]
    internvl = InternVLChatModel.from_pretrained(internvl_path)
    internvl = add_down_proj(internvl, config.model)
    
    ckpt_path = None

    internvl = internvl.to(device, dtype).eval()

    tokenizer = AutoTokenizer.from_pretrained(internvl_path, trust_remote_code=True, use_fast=False)