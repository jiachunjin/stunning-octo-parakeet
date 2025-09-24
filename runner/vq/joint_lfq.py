import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import copy
import torch
from transformers import AutoTokenizer

from util.trainer import Trainer
from util.vocab_expansion import expand_vocab_for_lfq, get_lfq_token_range
from util.parameter_stats import comprehensive_model_stats

def equip_internvl(internvl, config):
    from model.vq.lfq import LFQ_transformer
    
    # 扩展词汇表以支持LFQ tokens
    print("正在扩展词汇表以支持LFQ tokens...")
    internvl = expand_vocab_for_lfq(
        model                 = internvl,
        lfq_config            = config.down_proj,
        embedding_init_method = config.vocab_expansion.embedding_init_method,
        freeze_original       = config.vocab_expansion.freeze_original,
    )
    
    # 获取LFQ token范围
    start_token_id, end_token_id = get_lfq_token_range(internvl, config.down_proj)
    print(f"LFQ token范围: {start_token_id} - {end_token_id}")
    
    # add transformer vq
    lfq = LFQ_transformer(config.down_proj)
    lfq.requires_grad_(True)
    num_params = sum(p.numel() for p in lfq.parameters() if p.requires_grad)
    print(f"lfq 可训练参数量: {num_params}")
    internvl.lfq = lfq
    
    # 保存LFQ token信息
    internvl.lfq_start_token_id = start_token_id
    internvl.lfq_end_token_id = end_token_id
    # internvl.lfq_output_dim = config.down_proj.output_dim

    if config.tune_llm:
        internvl.language_model.model.requires_grad_(True)
        num_params = sum(p.numel() for p in internvl.language_model.model.parameters() if p.requires_grad)
        print(f"tune_llm: True, 可训练参数量: {num_params}")
    else:
        internvl.language_model.model.requires_grad_(False)
        print(f"tune_llm: False")

    return internvl

class MyTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def _load_models(self):
        from model.internvl.modeling_internvl_chat import InternVLChatModel

        internvl = InternVLChatModel.from_pretrained(self.config.model.internvl_path)
        internvl.requires_grad_(False)
        teacher = copy.deepcopy(internvl)
        teacher.requires_grad_(False)

        internvl = equip_internvl(internvl, self.config.model)
        if self.config.train.resume_path is not None:
            ckpt = torch.load(self.config.train.resume_path, map_location="cpu", weights_only=True)
            ckpt = {k: v for k, v in ckpt.items() if k not in self.config.train.skip_keys}
            m, u = internvl.load_state_dict(ckpt, strict=False)
            print(f"missing keys: {m}, unmatched keys: {u}")

        self.teacher = teacher.to(self.device, self.dtype).eval()
        self.model = internvl
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.internvl_path, trust_remote_code=True, use_fast=False)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

        if self.accelerator.is_main_process:
            print()
            print("="*80)
            num_trainable_params = sum(p.numel() for p in self.model.vision_model.parameters() if p.requires_grad)
            print(f"vision model 可训练参数量: {num_trainable_params / 1e6:.2f}M")
            num_trainable_params = sum(p.numel() for p in self.model.language_model.parameters() if p.requires_grad)
            print(f"language_model 可训练参数量: {num_trainable_params / 1e6:.2f}M")
            num_trainable_params = sum(p.numel() for p in self.model.mlp1.parameters() if p.requires_grad)
            print(f"mlp1 可训练参数量: {num_trainable_params / 1e6:.2f}M")
            num_trainable_params = sum(p.numel() for p in self.model.lfq.parameters() if p.requires_grad)
            print(f"lfq 可训练参数量: {num_trainable_params / 1e6:.2f}M")
            # num_trainable_params = sum(p.numel() for p in self.model.get_input_embeddings().parameters() if p.requires_grad)
            # print(f"input_embeddings 可训练参数量: {num_trainable_params / 1e6:.2f}M")
            # num_trainable_params = sum(p.numel() for p in self.model.get_output_embeddings().parameters() if p.requires_grad)
            # print(f"output_embeddings 可训练参数量: {num_trainable_params / 1e6:.2f}M")
    
    def _load_dataloader(self):
        from util.dataloader import get_blip3o_dataloader
        from util.dataloader import get_llava_mix665k_dataloader

        self.dataloader_gen = get_blip3o_dataloader(self.config.data.gen, self.accelerator)
        self.dataloader_und = get_llava_mix665k_dataloader(self.config.data.und)
    
    def train(self):
        self.model, self.optimizer, self.dataloader_und = self.accelerator.prepare(self.model, self.optimizer, self.dataloader_und)

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        imagenet_mean = torch.tensor(IMAGENET_MEAN, device=self.accelerator.device, dtype=self.dtype).view(1, 3, 1, 1)
        imagenet_std = torch.tensor(IMAGENET_STD, device=self.accelerator.device, dtype=self.dtype).view(1, 3, 1, 1)

        training_done = False

        und_iter = iter(self.dataloader_und)
        while not training_done:
            for batch_gen in self.dataloader_gen:
                with self.accelerator.accumulate(self.model):
                    self.model.train()

                    try:
                        batch_und = next(und_iter)
                    except StopIteration:
                        und_iter = iter(self.dataloader_und)
                        batch_und = next(und_iter)

                    pixel_values_gen = batch_gen["pixel_values"].to(self.device, self.dtype)
                    input_ids_gen = batch_gen["input_ids"].to(self.device)
                    attention_mask_gen = batch_gen["attention_mask"].to(self.device)
                    x_gen = (pixel_values_gen - imagenet_mean) / imagenet_std

                    x_und = batch_und["pixel_values"].to(self.dtype)
                    input_ids_und = batch_und["input_ids"].to(torch.int64)
                    attention_mask_und = batch_und["attention_mask"].to(torch.bool)
                    answer_mask_und = batch_und["answer_mask"].to(torch.bool)

                    B_gen, B_und = x_gen.shape[0], x_und.shape[0]

                    # ---------- get vit features ----------
                    with torch.no_grad():
                        vit_features = self.model.get_vit_feature(torch.cat([x_gen, x_und], dim=0))

                    x_vq, code = self.model.lfq(vit_features)
                    x_vq_gen = x_vq[:B_gen]
                    x_vq_und = x_vq[B_gen:]
                    vit_features_gen = vit_features[:B_gen]
                    vit_features_und = vit_features[B_gen:]

                    # ---------- understanding distillation ----------
                    vit_embeds_teacher = self.teacher.mlp1(vit_features_und)
                    # build input embeddings for teacher and model
                    input_embeds_teacher = self.teacher.language_model.get_input_embeddings()(input_ids_und)
                    B, N, C = input_embeds_teacher.shape
                    input_embeds_teacher = input_embeds_teacher.reshape(B * N, C)

                    input_ids_und = input_ids_und.reshape(B * N)
                    selected = (input_ids_und == self.img_context_token_id)
                    assert selected.sum() != 0
                    input_embeds_student = input_embeds_teacher.clone()
                    input_embeds_student[selected] = x_vq_und.reshape(-1, C).to(input_embeds_student.device)
                    input_embeds_teacher[selected] = vit_embeds_teacher.reshape(-1, C).to(input_embeds_teacher.device)

                    input_embeds_student = input_embeds_student.reshape(B, N, C)
                    input_embeds_teacher = input_embeds_teacher.reshape(B, N, C)

                    # compute understanding distillation loss
                    answer_logits_student = self.model.language_model(
                        inputs_embeds        = torch.cat([input_embeds_teacher, input_embeds_student], dim=0),
                        attention_mask       = torch.cat([attention_mask_und, attention_mask_und], dim=0),
                        output_hidden_states = False,
                    ).logits[answer_mask_und.repeat(2, 1)][:, :self.model.lfq_start_token_id]

                    answer_logits_teacher = self.teacher.language_model(
                        inputs_embeds        = input_embeds_teacher,
                        attention_mask       = attention_mask_und,
                        output_hidden_states = False,
                    ).logits[answer_mask_und]

                    answer_logits_teacher = answer_logits_teacher
                    answer_logits_student_complete_feature = answer_logits_student[:answer_logits_student.shape[0]//2]
                    answer_logits_student_vq_feature = answer_logits_student[answer_logits_student.shape[0]//2:]

                    # print(answer_logits_student.shape)
                    # print(answer_logits_teacher.shape)

                    answer_logits_student_log_softmax_complete_feature = torch.nn.functional.log_softmax(answer_logits_student_complete_feature, dim=-1)
                    answer_logits_teacher_log_softmax_vq_feature = torch.nn.functional.log_softmax(answer_logits_student_vq_feature, dim=-1)
                    answer_logits_teacher_log_softmax = torch.nn.functional.log_softmax(answer_logits_teacher, dim=-1)

                    kl_div_complete = torch.nn.functional.kl_div(answer_logits_student_log_softmax_complete_feature, answer_logits_teacher_log_softmax, log_target=True, reduction='batchmean')

                    kl_div_vq = torch.nn.functional.kl_div(answer_logits_teacher_log_softmax_vq_feature, answer_logits_teacher_log_softmax, log_target=True, reduction='batchmean')

                    self.accelerator.print(kl_div_complete, kl_div_vq)
                    # self.accelerator.print(vit_features_gen.shape, vit_features_und.shape)
                    exit(0)
                    




def main(args):
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config)
    trainer = MyTrainer(config)
    trainer.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)