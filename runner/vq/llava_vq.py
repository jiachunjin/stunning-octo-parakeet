import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import copy
import torch
import argparse
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from model.internvl.modeling_internvl_chat import InternVLChatModel
from util.trainer import Trainer
from util.dataloader import get_llava_mix665k_dataloader

from model.vq.lfq import LFQ, LFQ_transformer

def add_vq(internvl, config):
    if config.tune_llm:
        internvl.language_model.requires_grad_(True)
        print(f"tune_llm: True")
    else:
        internvl.language_model.requires_grad_(False)
        print(f"tune_llm: False")

    internvl.requires_grad_(False)
    if getattr(config, "down_proj_type", "linear") == "linear":
        lfq = LFQ(config)
    elif getattr(config, "down_proj_type", "linear") == "transformer":
        lfq = LFQ_transformer(config)
    else:
        raise ValueError(f"Invalid down_proj_type: {config.down_proj_type}")
    num_params = sum(p.numel() for p in lfq.parameters() if p.requires_grad)
    print(f"lfq 可训练参数量: {num_params}")
    lfq.requires_grad_(True)
    internvl.lfq = lfq

    return internvl

class MyTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def _load_models(self):
        internvl = InternVLChatModel.from_pretrained(self.config.model.internvl_path)
        internvl.requires_grad_(False)
        teacher = copy.deepcopy(internvl)
        internvl = add_vq(internvl, self.config.model)
        if self.config.train.resume_path is not None:
            ckpt = torch.load(self.config.train.resume_path, map_location="cpu", weights_only=True)
            m, u = internvl.load_state_dict(ckpt, strict=False)
            print(f"missing keys: {m}, unmatched keys: {u}")
        self.model = internvl
        self.teacher = teacher.to(self.device, self.dtype).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.internvl_path, trust_remote_code=True, use_fast=False)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    def _load_dataloader(self):
        self.dataloader_und = get_llava_mix665k_dataloader(self.config.data)

    def train(self):
        self.model, self.optimizer, self.dataloader_und = self.accelerator.prepare(self.model, self.optimizer, self.dataloader_und)
        training_done = False

        while not training_done:
            for batch in self.dataloader_und:
                with self.accelerator.accumulate(self.model):
                    self.model.train()
                    if batch is None:
                        continue

                    pixel_values_und = batch["pixel_values"].to(self.dtype)
                    input_ids_und = batch["input_ids"].to(torch.int64)
                    attention_mask_und = batch["attention_mask"].to(torch.bool)
                    answer_mask_und = batch["answer_mask"].to(torch.bool)

                    # get visual feature
                    vit_feature = self.model.get_vit_feature(pixel_values_und)
                    x_vq, code = self.model.lfq(vit_feature)
                    vit_embeds_teacher = self.teacher.mlp1(vit_feature)
                    
                    # build input embeddings for teacher and model
                    input_embeds_teacher = self.teacher.language_model.get_input_embeddings()(input_ids_und)
                    B, N, C = input_embeds_teacher.shape
                    input_embeds_teacher = input_embeds_teacher.reshape(B * N, C)

                    input_ids_und = input_ids_und.reshape(B * N)
                    selected = (input_ids_und == self.img_context_token_id)
                    assert selected.sum() != 0
                    input_embeds_student = input_embeds_teacher.clone()
                    input_embeds_student[selected] = x_vq.reshape(-1, C).to(input_embeds_student.device)
                    input_embeds_teacher[selected] = vit_embeds_teacher.reshape(-1, C).to(input_embeds_teacher.device)

                    input_embeds_student = input_embeds_student.reshape(B, N, C)
                    input_embeds_teacher = input_embeds_teacher.reshape(B, N, C)

                    # compute understanding distillation loss
                    answer_logits_student = self.model.language_model(
                        inputs_embeds        = input_embeds_student,
                        attention_mask       = attention_mask_und,
                        output_hidden_states = False,
                    ).logits[answer_mask_und]

                    answer_logits_teacher = self.teacher.language_model(
                        inputs_embeds        = input_embeds_teacher,
                        attention_mask       = attention_mask_und,
                        output_hidden_states = False,
                    ).logits[answer_mask_und]

                    answer_logits_student_log_softmax = torch.nn.functional.log_softmax(answer_logits_student, dim=-1)
                    answer_logits_teacher_log_softmax = torch.nn.functional.log_softmax(answer_logits_teacher, dim=-1)
                    kl_div = torch.nn.functional.kl_div(answer_logits_student_log_softmax, answer_logits_teacher_log_softmax, log_target=True, reduction='batchmean')

                    loss_und = kl_div

                    # back propogation
                    self.accelerator.backward(loss_und)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.params_to_learn, 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        self.global_step += 1
                        self.progress_bar.update(1)
                        logs = dict(
                            loss_und = self.accelerator.gather(loss_und.detach()).mean().item(),
                        )
                        self.accelerator.log(logs, step=self.global_step)
                        self.progress_bar.set_postfix(**logs)

                    if self.global_step > 0 and self.global_step % self.config.train.save_every == 0 and self.accelerator.is_main_process:
                        self.model.eval()
                        state_dict = self.accelerator.unwrap_model(self.model).state_dict()
                        save_path = os.path.join(self.output_dir, f"internvl-{self.config.train.exp_name}-{self.global_step}")
                        torch.save(state_dict, save_path)
                        print(f"internvl saved to {save_path}")

                    self.accelerator.wait_for_everyone()

            self.epoch += 1
            self.accelerator.print(f"epoch {self.epoch}: finished")
            self.accelerator.log({"epoch": self.epoch}, step=self.global_step)

        self.accelerator.end_training()



def main(args):
    config = OmegaConf.load(args.config)
    trainer = MyTrainer(config)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)