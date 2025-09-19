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
from einops import rearrange
from omegaconf import OmegaConf
from diffusers import DDPMScheduler
from transformers import AutoTokenizer

from util.trainer import Trainer
from util.dataloader import get_llava_mix665k_dataloader, get_blip3o_dataloader
from model.internvl.modeling_internvl_chat import InternVLChatModel
from model.diff_mlp import SimpleMLPAdaLN

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def pixel_shuffle(x, scale_factor=0.5):
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                int(c / (scale_factor * scale_factor)))

    x = x.permute(0, 2, 1, 3).contiguous()

    return x

def extract_feature_pre_adapter(vision_model, pixel_values):
    vit_embeds = vision_model(
        pixel_values         = pixel_values,
        output_hidden_states = False,
        return_dict          = True
    ).last_hidden_state

    vit_embeds = vit_embeds[:, 1:, :]
    
    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
    vit_embeds = pixel_shuffle(vit_embeds, scale_factor=0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

    return vit_embeds

def equip_internvl(internvl, config):
    internvl.requires_grad_(False)
    down_proj = torch.nn.Linear(config.high_dim, config.down_dim)
    new_mlp1 = torch.nn.Linear(config.down_dim, 1024)
    new_mlp2 = torch.nn.Linear(config.down_dim, 1024)
    diff_head = SimpleMLPAdaLN(
        in_channels    = config.diffhead.x_dim,
        model_channels = config.diffhead.hidden_size,
        out_channels   = config.diffhead.x_dim,
        z_channels     = config.diffhead.z_dim,
        num_res_blocks = config.diffhead.depth,
    )
    down_proj.requires_grad_(True)
    new_mlp1.requires_grad_(True)
    new_mlp2.requires_grad_(True)
    diff_head.requires_grad_(True)
    
    # num_params = sum(p.numel() for p in down_proj.parameters())
    # print(f"[down_proj] number of parameters: {num_params / 1e6} M")
    # num_params = sum(p.numel() for p in new_mlp1.parameters())
    # print(f"[new_mlp1] number of parameters: {num_params / 1e6} M")
    # num_params = sum(p.numel() for p in diff_head.parameters())
    # print(f"[diff_head] number of parameters: {num_params / 1e6} M")

    
    internvl.down_proj = down_proj
    internvl.new_mlp1 = new_mlp1
    internvl.new_mlp2 = new_mlp2
    internvl.diff_head = diff_head

    return internvl

class MyTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
    
    def _load_models(self):
        internvl = InternVLChatModel.from_pretrained(self.config.model.internvl_path)
        teacher = copy.deepcopy(internvl)

        internvl = equip_internvl(internvl, self.config.model)
        if self.config.train.resume_path is not None:
            ckpt = torch.load(self.config.train.resume_path, map_location="cpu", weights_only=True)
            m, u = internvl.load_state_dict(ckpt, strict=False)
            print(f"missing keys: {m}, unmatched keys: {u}")

        teacher.requires_grad_(False)
        teacher = teacher.to(self.device, self.dtype).eval()
        self.teacher = teacher
        self.model = internvl

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.internvl_path, trust_remote_code=True, use_fast=False)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    def _load_dataloader(self):
        self.dataloader_und = get_llava_mix665k_dataloader(self.config.data.und)
        self.dataloader_gen = get_blip3o_dataloader(self.config.data.gen, self.accelerator)
    
    def train(self):
        train_scheduler = DDPMScheduler(
            beta_schedule          = "scaled_linear",
            beta_start             = 0.00085,
            beta_end               = 0.012,
            num_train_timesteps    = 1000,
            clip_sample            = False,
            prediction_type        = "v_prediction",
            steps_offset           = 1,
            trained_betas          = None,
            timestep_spacing       = "trailing",
            rescale_betas_zero_snr = True
        )
        # do not need to prepare dataloader_gen
        self.model, self.optimizer, self.dataloader_und = self.accelerator.prepare(self.model, self.optimizer, self.dataloader_und)

        training_done = False

        imagenet_mean = torch.tensor(IMAGENET_MEAN, device=self.accelerator.device, dtype=self.dtype).view(1, 3, 1, 1)
        imagenet_std = torch.tensor(IMAGENET_STD, device=self.accelerator.device, dtype=self.dtype).view(1, 3, 1, 1)

        while not training_done:
            for batch in self.dataloader_gen:
                with self.accelerator.accumulate(self.model):
                    self.model.train()
                    # ---------- compute generation loss ----------
                    pixel_values = batch["pixel_values"].to(self.device, self.dtype)
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    x_intern = (pixel_values - imagenet_mean) / imagenet_std
                    with torch.no_grad():
                        vit_embeds = self.teacher.vision_model(
                            pixel_values         = x_intern,
                            output_hidden_states = False,
                        return_dict=True).last_hidden_state[:, 1:, :] # (B, 1024, 1024)

                        h = w = int(vit_embeds.shape[1] ** 0.5)
                        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
                        vit_embeds = self.teacher.pixel_shuffle(vit_embeds, scale_factor=self.teacher.downsample_ratio)
                        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1]) # (B, 256, 4096)

                    x_clip_16 = self.model.down_proj(vit_embeds)
                    
                    B = x_clip_16.shape[0]
                    img_embedding_gen = self.model.new_mlp2(x_clip_16) # (B, 256, d_llm)
                    text_embedding = self.model.language_model.get_input_embeddings()(input_ids).clone()
                    joint_embedding_t2i = torch.cat((text_embedding, img_embedding_gen), dim=1)
                    img_mask = torch.ones((B, self.config.data.gen.num_img_token), dtype=torch.bool, device=self.device)
                    attention_mask_t2i = torch.cat([attention_mask, img_mask], dim=1)

                    hidden_states = self.model.language_model(
                        inputs_embeds        = joint_embedding_t2i,
                        attention_mask       = attention_mask_t2i,
                        output_hidden_states = True,
                    ).hidden_states[-1]

                    hidden_state = hidden_states[:, -self.config.data.gen.num_img_token-1:-1, :]
                    z = rearrange(hidden_state, "B L D -> (B L) D")
                    gt_feature = rearrange(x_clip_16.detach(), "B L D -> (B L) D")
                    timesteps = torch.randint(0, 1000, (z.shape[0],), dtype=torch.int64, device=z.device)
                    noise = torch.randn_like(gt_feature, device=z.device, dtype=z.dtype)
                    noisy_latents = train_scheduler.add_noise(gt_feature, noise, timesteps)
                    target = train_scheduler.get_velocity(gt_feature, noise, timesteps)
                    pred = self.model.diff_head(noisy_latents, timesteps, z)

                    loss_gen = torch.nn.functional.mse_loss(pred, target)

                    # ---------- compute understanding distillation loss ----------
                    # 获取有效的batch，避免死循环
                    batch = None
                    for batch in self.dataloader_und:
                        if batch is not None:
                            break
                    pixel_values = batch["pixel_values"].to(self.dtype)
                    input_ids = batch["input_ids"].to(torch.int64)
                    attention_mask = batch["attention_mask"].to(torch.bool)
                    answer_mask = batch["answer_mask"].to(torch.bool)

                    

                    with torch.no_grad():
                        vit_embeds = self.teacher.vision_model(
                            pixel_values         = pixel_values,
                            output_hidden_states = False,
                        return_dict=True).last_hidden_state[:, 1:, :]

                        h = w = int(vit_embeds.shape[1] ** 0.5)
                        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
                        vit_embeds = self.teacher.pixel_shuffle(vit_embeds, scale_factor=self.teacher.downsample_ratio)
                        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

                        vit_embeds_teacher = self.teacher.mlp1(vit_embeds)

                    vit_embeds_student = self.model.new_mlp1(self.model.down_proj(vit_embeds))

                    input_embeds_teacher = self.teacher.language_model.get_input_embeddings()(input_ids)
                    B, N, C = input_embeds_teacher.shape
                    input_embeds_teacher = input_embeds_teacher.reshape(B * N, C)

                    input_ids = input_ids.reshape(B * N)
                    selected = (input_ids == self.img_context_token_id)
                    assert selected.sum() != 0
                    input_embeds_student = input_embeds_teacher.clone()
                    input_embeds_student[selected] = vit_embeds_student.reshape(-1, C).to(input_embeds_student.device)
                    input_embeds_teacher[selected] = vit_embeds_teacher.reshape(-1, C).to(input_embeds_teacher.device)

                    input_embeds_student = input_embeds_student.reshape(B, N, C)
                    input_embeds_teacher = input_embeds_teacher.reshape(B, N, C)

                    logits_student = self.model.language_model(
                        inputs_embeds        = input_embeds_student,
                        attention_mask       = attention_mask,
                        output_hidden_states = True,
                    ).logits
                    # 只取answer部分的logits出来
                    # 使用answer_mask来选择答案部分的logits
                    answer_logits_student = logits_student[answer_mask]

                    logits_teacher = self.teacher.language_model(
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

                    loss_und = kl_div

                    # ---------- backward the total loss ----------
                    loss = self.config.train.hp_loss_gen * loss_gen + self.config.train.hp_loss_und * loss_und

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.params_to_learn, 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        self.global_step += 1
                        self.progress_bar.update(1)
                        logs = dict(
                            loss_gen = self.accelerator.gather(loss_gen.detach()).mean().item(),
                            loss_und = self.accelerator.gather(loss_und.detach()).mean().item(),
                            loss = self.accelerator.gather(loss.detach()).mean().item(),
                        )
                        self.accelerator.log(logs, step=self.global_step)
                        self.progress_bar.set_postfix(**logs)


def main(args):
    config = OmegaConf.load(args.config)
    trainer = MyTrainer(config)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)