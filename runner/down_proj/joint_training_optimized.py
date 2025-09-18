"""
优化版本的联合训练代码
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
from util.dataloader import get_blip3o_dataloader
from util.dataloader_batched import get_llava_mix665k_dataloader_batched
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
        # 使用批量数据加载器，从配置中获取batch size
        batch_size_und = self.config.data.und.batch_size if hasattr(self.config.data.und, 'batch_size') else 4
        self.dataloader_und = get_llava_mix665k_dataloader_batched(
            batch_size=batch_size_und,
            num_workers=4
        )
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

        # 预先计算并移动到GPU的常量
        imagenet_mean = torch.tensor(IMAGENET_MEAN, device=self.accelerator.device, dtype=self.dtype).view(1, 3, 1, 1)
        imagenet_std = torch.tensor(IMAGENET_STD, device=self.accelerator.device, dtype=self.dtype).view(1, 3, 1, 1)
        
        # 创建迭代器以避免在训练循环中处理
        dataloader_und_iter = iter(self.dataloader_und)

        while not training_done:
            for batch_gen in self.dataloader_gen:
                with self.accelerator.accumulate(self.model):
                    self.model.train()
                    
                    # 获取understanding batch，如果迭代器结束则重新创建
                    try:
                        batch_und = next(dataloader_und_iter)
                        while batch_und is None:
                            batch_und = next(dataloader_und_iter)
                    except StopIteration:
                        dataloader_und_iter = iter(self.dataloader_und)
                        batch_und = next(dataloader_und_iter)
                    
                    # ---------- 批处理视觉特征提取 ----------
                    # 提前移动所有数据到GPU
                    pixel_values_gen = batch_gen["pixel_values"].to(self.device, self.dtype)
                    pixel_values_und = batch_und["pixel_values"].to(self.device, self.dtype)
                    
                    # 合并两个batch的图像，一次性处理
                    all_pixel_values = torch.cat([
                        (pixel_values_gen - imagenet_mean) / imagenet_std,
                        pixel_values_und
                    ], dim=0)
                    
                    with torch.no_grad():
                        # 一次性提取所有视觉特征
                        all_vit_embeds = self.teacher.vision_model(
                            pixel_values         = all_pixel_values,
                            output_hidden_states = False,
                            return_dict=True
                        ).last_hidden_state[:, 1:, :]
                        
                        # 批量处理pixel shuffle
                        h = w = int(all_vit_embeds.shape[1] ** 0.5)
                        all_vit_embeds = all_vit_embeds.reshape(all_vit_embeds.shape[0], h, w, -1)
                        all_vit_embeds = self.teacher.pixel_shuffle(all_vit_embeds, scale_factor=self.teacher.downsample_ratio)
                        all_vit_embeds = all_vit_embeds.reshape(all_vit_embeds.shape[0], -1, all_vit_embeds.shape[-1])
                        
                        # 分离生成和理解的特征
                        batch_size_gen = pixel_values_gen.shape[0]
                        vit_embeds_gen = all_vit_embeds[:batch_size_gen]
                        vit_embeds_und = all_vit_embeds[batch_size_gen:]
                        
                        # 计算teacher的理解特征
                        vit_embeds_teacher = self.teacher.mlp1(vit_embeds_und)
                    
                    # ---------- 计算生成损失 ----------
                    input_ids_gen = batch_gen["input_ids"].to(self.device)
                    attention_mask_gen = batch_gen["attention_mask"].to(self.device)
                    
                    x_clip_16 = self.model.down_proj(vit_embeds_gen)
                    
                    B = x_clip_16.shape[0]
                    img_embedding_gen = self.model.new_mlp2(x_clip_16)
                    text_embedding = self.model.language_model.get_input_embeddings()(input_ids_gen)
                    joint_embedding_t2i = torch.cat((text_embedding, img_embedding_gen), dim=1)
                    
                    # 使用torch.ones代替创建新的mask
                    img_mask = torch.ones((B, self.config.data.gen.num_img_token), 
                                        dtype=torch.bool, device=self.device)
                    attention_mask_t2i = torch.cat([attention_mask_gen, img_mask], dim=1)

                    hidden_states = self.model.language_model(
                        inputs_embeds        = joint_embedding_t2i,
                        attention_mask       = attention_mask_t2i,
                        output_hidden_states = True,
                    ).hidden_states[-1]

                    hidden_state = hidden_states[:, -self.config.data.gen.num_img_token-1:-1, :]
                    z = rearrange(hidden_state, "B L D -> (B L) D")
                    gt_feature = rearrange(x_clip_16.detach(), "B L D -> (B L) D")
                    
                    # 批量生成随机数
                    timesteps = torch.randint(0, 1000, (z.shape[0],), dtype=torch.int64, device=self.device)
                    noise = torch.randn_like(gt_feature)
                    
                    noisy_latents = train_scheduler.add_noise(gt_feature, noise, timesteps)
                    target = train_scheduler.get_velocity(gt_feature, noise, timesteps)
                    pred = self.model.diff_head(noisy_latents, timesteps, z)

                    loss_gen = torch.nn.functional.mse_loss(pred, target)

                    # ---------- 计算理解蒸馏损失 ----------
                    question = batch_und["question"]
                    answer = batch_und["answer"]
                    question_attention_mask = batch_und["question_attention_mask"]
                    answer_attention_mask = batch_und["answer_attention_mask"]

                    # 拼接question和answer
                    input_ids_und = torch.cat([question, answer], dim=1).to(self.device, torch.int64)
                    attention_mask_und = torch.cat([question_attention_mask, answer_attention_mask], dim=1).to(self.device)

                    # 计算student的理解特征
                    vit_embeds_student = self.model.new_mlp1(self.model.down_proj(vit_embeds_und))

                    # 获取语言模型嵌入
                    input_embeds_base = self.teacher.language_model.get_input_embeddings()(input_ids_und)
                    B, N, C = input_embeds_base.shape
                    
                    # 优化：避免深拷贝，使用clone和高效的索引操作
                    input_embeds_teacher = input_embeds_base.clone()
                    input_embeds_student = input_embeds_base.clone()
                    
                    # 找到需要替换的位置
                    input_ids_flat = input_ids_und.reshape(-1)
                    selected_mask = (input_ids_flat == self.img_context_token_id)
                    
                    if selected_mask.sum() > 0:
                        # 使用高效的scatter操作替换
                        selected_indices = selected_mask.nonzero(as_tuple=True)[0]
                        
                        # 计算替换的嵌入
                        vit_embeds_teacher_flat = vit_embeds_teacher.reshape(-1, C)
                        vit_embeds_student_flat = vit_embeds_student.reshape(-1, C)
                        
                        # 批量替换
                        input_embeds_teacher_flat = input_embeds_teacher.reshape(-1, C)
                        input_embeds_student_flat = input_embeds_student.reshape(-1, C)
                        
                        input_embeds_teacher_flat[selected_indices] = vit_embeds_teacher_flat
                        input_embeds_student_flat[selected_indices] = vit_embeds_student_flat
                        
                        input_embeds_teacher = input_embeds_teacher_flat.reshape(B, N, C)
                        input_embeds_student = input_embeds_student_flat.reshape(B, N, C)

                    # 并行计算student和teacher的logits
                    logits_student = self.model.language_model(
                        inputs_embeds        = input_embeds_student,
                        attention_mask       = attention_mask_und,
                        output_hidden_states = True,
                    ).logits

                    with torch.no_grad():
                        logits_teacher = self.teacher.language_model(
                            inputs_embeds        = input_embeds_teacher,
                            attention_mask       = attention_mask_und,
                            output_hidden_states = True,
                        ).logits

                    # 计算KL散度损失 - 仅在答案部分
                    # 获取答案部分的起始位置（question长度）
                    question_lengths = question_attention_mask.sum(dim=1)  # (B,)
                    answer_lengths = answer_attention_mask.sum(dim=1)      # (B,)
                    
                    # 初始化损失
                    kl_div_total = 0.0
                    valid_tokens = 0
                    
                    # 对每个样本单独处理，以正确定位答案部分
                    for i in range(input_ids_und.shape[0]):
                        q_len = question_lengths[i].item()
                        a_len = answer_lengths[i].item()
                        
                        if a_len > 0:  # 确保有答案
                            # 提取答案部分的logits (不包括最后一个token)
                            answer_logits_student = logits_student[i, q_len:q_len+a_len-1, :]
                            answer_logits_teacher = logits_teacher[i, q_len:q_len+a_len-1, :]
                            
                            # 计算该样本的KL散度
                            answer_log_softmax_student = torch.nn.functional.log_softmax(answer_logits_student, dim=-1)
                            answer_log_softmax_teacher = torch.nn.functional.log_softmax(answer_logits_teacher, dim=-1)
                            
                            kl_div = torch.nn.functional.kl_div(
                                answer_log_softmax_student,
                                answer_log_softmax_teacher,
                                log_target=True,
                                reduction='mean'
                            )
                            
                            kl_div_total += kl_div
                            valid_tokens += (a_len - 1)  # 答案长度减1（因为预测下一个token）
                    
                    # 平均KL散度
                    loss_und = kl_div_total / max(valid_tokens, 1)

                    # ---------- 反向传播总损失 ----------
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
