import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import copy
import torch
import argparse
import torch.nn as nn
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from tqdm import tqdm

from util.accelerator import get_accelerator
from util.dataloader import get_llava_mix665k_dataloader
from util.misc import flatten_dict
from model.internvl.modeling_internvl_chat import InternVLChatModel
from model.projector.transformer import Transformer_Projector

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def add_down_proj(internvl, config):
    internvl.requires_grad_(False)
    if config.proj_type == "linear":
        down_proj = nn.Linear(config.high_dim, config.down_dim)
        num_params = sum(p.numel() for p in down_proj.parameters() if p.requires_grad)
        print(f"down_proj 可训练参数量: {num_params}")
    elif config.proj_type == "transformer":
        down_proj = Transformer_Projector(config)
        num_params = sum(p.numel() for p in down_proj.parameters() if p.requires_grad)
        print(f"down_proj 可训练参数量: {num_params}")
    else:
        raise ValueError(f"Invalid proj_type: {config.proj_type}")

    internvl.down_proj = down_proj
    internvl.down_proj.requires_grad_(True)

    new_mlp1 = nn.Linear(config.down_dim, 1024)
    internvl.new_mlp1 = new_mlp1
    internvl.new_mlp1.requires_grad_(True)

    if getattr(config, "tune_llm", False):
        internvl.language_model.requires_grad_(True)
        print(f"tune_llm: True")

    return internvl

def main(args):
    config = OmegaConf.load(args.config)
    accelerator, output_dir = get_accelerator(config)

    internvl = InternVLChatModel.from_pretrained(config.model.internvl_path)
    teacher = copy.deepcopy(internvl)
    teacher.requires_grad_(False)
    
    internvl = add_down_proj(internvl, config.model)
    if config.train.resume_path is not None:
        ckpt = torch.load(config.train.resume_path, map_location="cpu", weights_only=True)
        m, u = internvl.load_state_dict(ckpt, strict=False)
        print(f"missing keys: {m}, unmatched keys: {u}")


    tokenizer = AutoTokenizer.from_pretrained(config.model.internvl_path, trust_remote_code=True, use_fast=False)
    img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    dataloader = get_llava_mix665k_dataloader(config.data)

    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(p for p in internvl.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = config.train.lr,
        betas        = (0.9, 0.95),
        weight_decay = 5e-2,
        eps          = 1e-8,
    )

    if accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    internvl, optimizer, dataloader = accelerator.prepare(internvl, optimizer, dataloader)
    teacher = teacher.to(accelerator.device, dtype).eval()

    training_done = False
    epoch = 0
    progress_bar = tqdm(
        total   = config.train.num_iter,
        initial = global_step,
        desc    = "Steps",
        disable = not accelerator.is_local_main_process,
    )

    config.device_count = accelerator.num_processes
    if accelerator.is_main_process:
        accelerator.init_trackers(config.train.wandb_proj, config=flatten_dict(config))
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            OmegaConf.save(config, f)

    accelerator.print(f"Learnable parameters: {sum(p.numel() for p in params_to_learn if p.requires_grad) / 1e6} M")
    accelerator.print(f"Accelerator mixed precision: {accelerator.mixed_precision}")
    
    while not training_done:
        for batch in dataloader:
            if batch is None:
                continue
            with accelerator.accumulate([internvl]):
                pixel_values = batch["pixel_values"].to(dtype)
                input_ids = batch["input_ids"].to(torch.int64)
                attention_mask = batch["attention_mask"].to(torch.bool)
                answer_mask = batch["answer_mask"].to(torch.bool)
                # question = batch["question"]
                # answer = batch["answer"]

                # answer_length = answer.shape[1]
                # input_ids = torch.cat([question, answer], dim=1).to(torch.int64)

                # construct input of the VLM
                with torch.no_grad():
                    vit_embeds = teacher.vision_model(
                        pixel_values         = pixel_values,
                        output_hidden_states = False,
                    return_dict=True).last_hidden_state[:, 1:, :]

                    h = w = int(vit_embeds.shape[1] ** 0.5)
                    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
                    vit_embeds = teacher.pixel_shuffle(vit_embeds, scale_factor=teacher.downsample_ratio)
                    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

                    vit_embeds_teacher = teacher.mlp1(vit_embeds)

                vit_embeds_student = internvl.new_mlp1(internvl.down_proj(vit_embeds))

                input_embeds_teacher = teacher.language_model.get_input_embeddings()(input_ids)
                B, N, C = input_embeds_teacher.shape
                input_embeds_teacher = input_embeds_teacher.reshape(B * N, C)

                input_ids = input_ids.reshape(B * N)
                selected = (input_ids == img_context_token_id)
                assert selected.sum() != 0
                input_embeds_student = input_embeds_teacher.clone()
                input_embeds_student[selected] = vit_embeds_student.reshape(-1, C).to(input_embeds_student.device)
                input_embeds_teacher[selected] = vit_embeds_teacher.reshape(-1, C).to(input_embeds_teacher.device)

                input_embeds_student = input_embeds_student.reshape(B, N, C)
                input_embeds_teacher = input_embeds_teacher.reshape(B, N, C)

                print(pixel_values.shape)
                print(input_embeds_student.shape)

                logits_student = internvl.language_model(
                    inputs_embeds        = input_embeds_student,
                    attention_mask       = attention_mask,
                    output_hidden_states = True,
                ).logits
                # 只取answer部分的logits出来
                # 使用answer_mask来选择答案部分的logits
                answer_logits_student = logits_student[answer_mask]

                logits_teacher = teacher.language_model(
                    inputs_embeds        = input_embeds_teacher,
                    attention_mask       = attention_mask,
                    output_hidden_states = True,
                ).logits
                # 只取answer部分的logits出来
                answer_logits_teacher = logits_teacher[answer_mask]

                # compute loss for the answer part
                # 使用answer部分的logits计算KL散度
                answer_logits_student_log_softmax = torch.nn.functional.log_softmax(answer_logits_student, dim=-1)
                answer_logits_teacher_log_softmax = torch.nn.functional.log_softmax(answer_logits_teacher, dim=-1)
                kl_div = torch.nn.functional.kl_div(answer_logits_student_log_softmax, answer_logits_teacher_log_softmax, log_target=True, reduction='batchmean')

                loss = kl_div

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    global_step += 1
                    progress_bar.update(1)
                    logs = dict(
                        loss_und = accelerator.gather(kl_div.detach()).mean().item(),
                    )
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)

                    if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                        internvl.eval()
                        state_dict = accelerator.unwrap_model(internvl).state_dict()
                        save_path = os.path.join(output_dir, f"internvl-{config.train.exp_name}-{global_step}")
                        torch.save(state_dict, save_path)
                        print(f"internvl saved to {save_path}")

                    accelerator.wait_for_everyone()

        epoch += 1
        accelerator.print(f"epoch {epoch}: finished")
        accelerator.log({"epoch": epoch}, step=global_step)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/down_proj/llava_distill.yaml")
    args = parser.parse_args()
    main(args)