import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from model.internvl.modeling_internvl_chat import InternVLChatModel
from runner.down_proj.llava_distill import add_down_proj
from runner.temp.ori_model_try import load_image
from datasets import load_dataset
from runner.temp.ori_model_try import extract_yes_no_answer


@torch.inference_mode()
def test_ori_down_proj():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # load trained internvl with new projector
    exp_dir = "/data/phd/jinjiachun/experiment/down_proj/0916_llava_distill_linear_debug"
    exp_name = exp_dir.split("/")[-1]
    step = 45000
    config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))
    internvl_path = "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3_5-1B"
    internvl = InternVLChatModel.from_pretrained(internvl_path)
    internvl = add_down_proj(internvl, config.model)
    
    ckpt_path = f"/data/phd/jinjiachun/experiment/down_proj/0916_llava_distill_linear_debug/internvl-down_proj-{step}"
    print(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m, u = internvl.load_state_dict(ckpt, strict=False)
    print(f"missing keys: {m}, unmatched keys: {u}")    

    internvl = internvl.to(device, dtype).eval()

    # do inference
    tokenizer = AutoTokenizer.from_pretrained(internvl_path, trust_remote_code=True, use_fast=False)

    data_files = {
        "test": "/data/phd/jinjiachun/dataset/benchmark/darkyarding/MME/data/test-*-of-*.parquet"
    }
    dataset = load_dataset("parquet", data_files=data_files)

    for i, data in enumerate(dataset["test"]):
        img_name = data["question_id"].split("/")[-1]
        category = data["category"]
        image = data["image"].convert("RGB")
        question = data["question"]
        gt_answer = data["answer"]

        pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()

        question_prime = '<image>\n' + question

        generation_config = dict(max_new_tokens=50, do_sample=False)

        # construct visual features
        vit_embeds = internvl.vision_model(
            pixel_values         = pixel_values,
            output_hidden_states = False,
        return_dict=True).last_hidden_state[:, 1:, :]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = internvl.pixel_shuffle(vit_embeds, scale_factor=internvl.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

        visual_features = internvl.new_mlp1(internvl.down_proj(vit_embeds))

        generation_config["visual_features"] = visual_features

        response_raw = internvl.chat(tokenizer, pixel_values, question_prime, generation_config)

        answer = extract_yes_no_answer(response_raw)
        print(response_raw, answer)
        model_name = ckpt_path.split("/")[-1]
        os.makedirs(f"evaluation/understanding/mme/{model_name}", exist_ok=True)
        with open(f"evaluation/understanding/mme/{model_name}/{category}.txt", "a") as f:
            line = f"{img_name}\t{question}\t{gt_answer}\t{answer}\n"
            f.write(line)

@torch.no_grad()
def img_describe():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # ---------- load trained internvl with new projector ----------
    config = OmegaConf.load("/data/phd/jinjiachun/experiment/down_proj/0916_llava_distill_linear_debug/config.yaml")
    internvl_path = "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3_5-1B"
    internvl = InternVLChatModel.from_pretrained(internvl_path)
    internvl = add_down_proj(internvl, config.model)
    
    ckpt_path = "/data/phd/jinjiachun/experiment/down_proj/0916_llava_distill_linear_debug/internvl-down_proj-45000"
    print(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m, u = internvl.load_state_dict(ckpt, strict=False)
    print(f"missing keys: {m}, unmatched keys: {u}")    

    internvl = internvl.to(device, dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(internvl_path, trust_remote_code=True, use_fast=False)

    # ---------- chat with the model ----------
    image = "/data/phd/jinjiachun/codebase/ideal-octo-spork/asset/internet/messi_1.jpg"
    pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()

    question_prime = "<image>\n" + "Describe this image in great detail."
    generation_config = dict(max_new_tokens=128, do_sample=True)
    
    # extract visual features from pixel values
    vit_embeds = internvl.vision_model(
        pixel_values         = pixel_values,
        output_hidden_states = False,
    return_dict=True).last_hidden_state[:, 1:, :]
    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
    vit_embeds = internvl.pixel_shuffle(vit_embeds, scale_factor=internvl.downsample_ratio)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

    visual_features = internvl.new_mlp1(internvl.down_proj(vit_embeds))

    generation_config["visual_features"] = visual_features

    response_raw = internvl.chat(tokenizer, pixel_values, question_prime, generation_config)
    
    print(response_raw)

if __name__ == "__main__":
    # test_ori_down_proj()
    img_describe()