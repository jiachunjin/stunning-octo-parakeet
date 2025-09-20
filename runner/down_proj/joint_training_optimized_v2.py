"""
优化版本的联合训练脚本，提高GPU利用率
主要优化：
1. 异步数据加载
2. 并行处理两个任务
3. 减少同步等待
4. 优化内存使用
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
import threading
import queue
import time

from util.trainer import Trainer
from util.dataloader_optimized import get_optimized_llava_dataloader
from util.dataloader import get_blip3o_dataloader
from model.internvl.modeling_internvl_chat import InternVLChatModel
from model.diff_mlp import SimpleMLPAdaLN

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def pixel_shuffle(x, scale_factor=0.5):
    n, w, h, c = x.size()
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                int(c / (scale_factor * scale_factor)))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x

def extract_feature_pre_adapter(vision_model, pixel_values):
    vit_embeds = vision_model(
        pixel_values         = pixel_values,
        output_hidden_states = False,
        return_dict=True).last_hidden_state[:, 1:, :]

    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
    vit_embeds = pixel_shuffle(vit_embeds, scale_factor=0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

    return vit_embeds

def equip_internvl(internvl, config):
    internvl.down_proj = torch.nn.Linear(config.high_dim, config.down_dim, bias=False)
    internvl.new_mlp1 = torch.nn.Linear(config.down_dim, 1024, bias=False)
    internvl.new_mlp2 = torch.nn.Linear(config.down_dim, 1024, bias=False)
    internvl.diff_head = SimpleMLPAdaLN(
        x_dim=config.diffhead.x_dim,
        hidden_size=config.diffhead.hidden_size,
        z_dim=config.diffhead.z_dim,
        depth=config.diffhead.depth,
    )
    return internvl

class OptimizedJointTrainer(Trainer):
    """优化的联合训练器"""
    
    def __init__(self, config):
        super().__init__(config)
        self.und_batch_queue = queue.Queue(maxsize=2)
        self.und_thread = None
        self.stop_und_thread = threading.Event()
    
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
        # 使用优化的数据加载器
        self.dataloader_und = get_optimized_llava_dataloader(self.config.data.und)
        self.dataloader_gen = get_blip3o_dataloader(self.config.data.gen, self.accelerator)
    
    def _start_und_data_loader(self):
        """启动理解任务数据加载线程"""
        def und_loader_worker():
            try:
                for batch in self.dataloader_und:
                    if self.stop_und_thread.is_set():
                        break
                    if batch is not None:
                        try:
                            self.und_batch_queue.put(batch, timeout=1)
                        except queue.Full:
                            # 如果队列满了，丢弃最老的batch
                            try:
                                self.und_batch_queue.get_nowait()
                                self.und_batch_queue.put(batch, timeout=1)
                            except queue.Empty:
                                pass
            except Exception as e:
                print(f"UND data loader error: {e}")
        
        self.und_thread = threading.Thread(target=und_loader_worker)
        self.und_thread.daemon = True
        self.und_thread.start()
    
    def _get_und_batch(self):
        """获取理解任务的batch，非阻塞"""
        try:
            return self.und_batch_queue.get(timeout=0.1)
        except queue.Empty:
            return None
    
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
        
        # 准备模型和优化器
        self.model, self.optimizer, self.dataloader_und = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader_und
        )

        # 启动理解任务数据加载线程
        self._start_und_data_loader()

        training_done = False
        imagenet_mean = torch.tensor(IMAGENET_MEAN, device=self.accelerator.device, dtype=self.dtype).view(1, 3, 1, 1)
        imagenet_std = torch.tensor(IMAGENET_STD, device=self.accelerator.device, dtype=self.dtype).view(1, 3, 1, 1)

        # 预加载一个理解任务的batch
        und_batch = self._get_und_batch()

        while not training_done:
            for batch in self.dataloader_gen:
                with self.accelerator.accumulate(self.model):
                    self.model.train()
                    
                    # ---------- 生成任务计算 ----------
                    pixel_values = batch["pixel_values"].to(self.device, self.dtype)
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    x_intern = (pixel_values - imagenet_mean) / imagenet_std
                    
                    with torch.no_grad():
                        vit_embeds = self.teacher.vision_model(
                            pixel_values         = x_intern,
                            output_hidden_states = False,
                        return_dict=True).last_hidden_state[:, 1:, :]

                        h = w = int(vit_embeds.shape[1] ** 0.5)
                        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
                        vit_embeds = self.teacher.pixel_shuffle(vit_embeds, scale_factor=self.teacher.downsample_ratio)
                        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

                    x_clip_16 = self.model.down_proj(vit_embeds)
                    
                    B = x_clip_16.shape[0]
                    img_embedding_gen = self.model.new_mlp2(x_clip_16)
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

                    # ---------- 理解任务计算（使用预加载的batch） ----------
                    loss_und = torch.tensor(0.0, device=self.device)
                    if und_batch is not None:
                        pixel_values_und = und_batch["pixel_values"].to(self.dtype)
                        input_ids_und = und_batch["input_ids"].to(torch.int64)
                        attention_mask_und = und_batch["attention_mask"].to(torch.bool)
                        answer_mask_und = und_batch["answer_mask"].to(torch.bool)

                        with torch.no_grad():
                            vit_embeds_und = self.teacher.vision_model(
                                pixel_values         = pixel_values_und,
                                output_hidden_states = False,
                            return_dict=True).last_hidden_state[:, 1:, :]

                            h = w = int(vit_embeds_und.shape[1] ** 0.5)
                            vit_embeds_und = vit_embeds_und.reshape(vit_embeds_und.shape[0], h, w, -1)
                            vit_embeds_und = self.teacher.pixel_shuffle(vit_embeds_und, scale_factor=self.teacher.downsample_ratio)
                            vit_embeds_und = vit_embeds_und.reshape(vit_embeds_und.shape[0], -1, vit_embeds_und.shape[-1])

                            vit_embeds_teacher = self.teacher.mlp1(vit_embeds_und)

                        vit_embeds_student = self.model.new_mlp1(self.model.down_proj(vit_embeds_und))

                        input_embeds_teacher = self.teacher.language_model.get_input_embeddings()(input_ids_und)
                        B_und, N, C = input_embeds_teacher.shape
                        input_embeds_teacher = input_embeds_teacher.reshape(B_und * N, C)

                        input_ids_und = input_ids_und.reshape(B_und * N)
                        selected = (input_ids_und == self.img_context_token_id)
                        if selected.sum() != 0:
                            input_embeds_student = input_embeds_teacher.clone()
                            input_embeds_student[selected] = vit_embeds_student.reshape(-1, C).to(input_embeds_student.device)
                            input_embeds_teacher[selected] = vit_embeds_teacher.reshape(-1, C).to(input_embeds_teacher.device)

                            input_embeds_student = input_embeds_student.reshape(B_und, N, C)
                            input_embeds_teacher = input_embeds_teacher.reshape(B_und, N, C)

                            logits_student = self.model.language_model(
                                inputs_embeds        = input_embeds_student,
                                attention_mask       = attention_mask_und,
                                output_hidden_states = True,
                            ).logits
                            
                            answer_logits_student = logits_student[answer_mask_und]

                            logits_teacher = self.teacher.language_model(
                                inputs_embeds        = input_embeds_teacher,
                                attention_mask       = attention_mask_und,
                                output_hidden_states = True,
                            ).logits
                            
                            answer_logits_teacher = logits_teacher[answer_mask_und]

                            answer_logits_student_log_softmax = torch.nn.functional.log_softmax(answer_logits_student, dim=-1)
                            answer_logits_teacher_log_softmax = torch.nn.functional.log_softmax(answer_logits_teacher, dim=-1)
                            loss_und = torch.nn.functional.kl_div(
                                answer_logits_student_log_softmax, 
                                answer_logits_teacher_log_softmax, 
                                log_target=True, 
                                reduction='batchmean'
                            )

                    # ---------- 反向传播 ----------
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

                    # 预加载下一个理解任务的batch
                    und_batch = self._get_und_batch()

                    if self.global_step >= self.config.train.num_iter:
                        training_done = True
                        break

        # 停止理解任务数据加载线程
        self.stop_und_thread.set()
        if self.und_thread:
            self.und_thread.join(timeout=5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/down_proj/joint_training.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    trainer = OptimizedJointTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()

