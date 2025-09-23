import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import argparse
from omegaconf import OmegaConf

from util.trainer import Trainer
from util.dataloader import get_blip3o_dataloader
from model.internvl.modeling_internvl_chat import InternVLChatModel


class MyTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def _load_models(self):
        from model.vq.lfq import LFQ_autoencoder
        self.model = LFQ_autoencoder(self.config.model)
        self.internvl = InternVLChatModel.from_pretrained(self.config.model.internvl_path)

        self.internvl = self.internvl.to(self.device, self.dtype).eval()


    def _load_dataloader(self):
        self.dataloader_gen = get_blip3o_dataloader(self.config.data, self.accelerator)
    
    def train(self):
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        training_done = False

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        imagenet_mean = torch.tensor(IMAGENET_MEAN, device=self.accelerator.device, dtype=self.dtype).view(1, 3, 1, 1)
        imagenet_std = torch.tensor(IMAGENET_STD, device=self.accelerator.device, dtype=self.dtype).view(1, 3, 1, 1)

        while not training_done:
            for batch in self.dataloader_gen:
                with self.accelerator.accumulate(self.model):
                    self.model.train()
                    pixel_values_gen = batch["pixel_values"].to(self.device, self.dtype)
                    x = (pixel_values_gen - imagenet_mean) / imagenet_std

                    vit_feature = self.internvl.get_vit_feature(x)
                    x_recon, code = self.model(vit_feature)

                    # compute cosine similarity between x_recon and x
                    cosine_similarity = torch.nn.functional.cosine_similarity(vit_feature, x_recon, dim=-1)
                    
                    loss = 1 - cosine_similarity.mean()

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.params_to_learn, 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        self.global_step += 1
                        self.progress_bar.update(1)
                        logs = dict(
                            loss_cosine = self.accelerator.gather(loss.detach()).mean().item(),
                        )
                        self.accelerator.log(logs, step=self.global_step)
                        self.progress_bar.set_postfix(**logs)

                    if self.global_step > 0 and self.global_step % self.config.train.save_every == 0 and self.accelerator.is_main_process:
                        self.model.eval()
                        state_dict = self.accelerator.unwrap_model(self.model).state_dict()
                        save_path = os.path.join(self.output_dir, f"autoencoder-{self.config.train.exp_name}-{self.global_step}")
                        torch.save(state_dict, save_path)
                        print(f"autoencoder saved to {save_path}")

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