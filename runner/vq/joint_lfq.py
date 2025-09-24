import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from util.trainer import Trainer
from util.vocab_expansion import expand_vocab_for_lfq, get_lfq_token_range

def equip_internvl(internvl, config):
    from model.vq.lfq import LFQ_transformer
    
    # 扩展词汇表以支持LFQ tokens
    print("正在扩展词汇表以支持LFQ tokens...")
    internvl = expand_vocab_for_lfq(
        model=internvl,
        lfq_config=config.down_proj,
        embedding_init_method="random"
    )
    
    # 获取LFQ token范围
    start_token_id, end_token_id = get_lfq_token_range(internvl, config.down_proj)
    print(f"LFQ token范围: {start_token_id} - {end_token_id}")
    
    # add transformer vq
    lfq = LFQ_transformer(config.down_proj)
    num_params = sum(p.numel() for p in lfq.parameters() if p.requires_grad)
    print(f"lfq 可训练参数量: {num_params}")
    lfq.requires_grad_(True)
    internvl.lfq = lfq
    
    # 保存LFQ token信息
    internvl.lfq_start_token_id = start_token_id
    internvl.lfq_end_token_id = end_token_id
    internvl.lfq_output_dim = config.down_proj.output_dim

    if config.model.tune_llm:
        internvl.language_model.requires_grad_(True)
        print(f"tune_llm: True")
    else:
        internvl.language_model.requires_grad_(False)
        print(f"tune_llm: False")

class MyTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def _load_models(self):
        from model.vq.lfq import LFQ_autoencoder
        from model.internvl.modeling_internvl_chat import InternVLChatModel

        self.down_proj = LFQ_autoencoder(self.config.model.down_proj)
        internvl = InternVLChatModel.from_pretrained(self.config.model.internvl_path)
        self.internvl = equip_internvl(internvl, self.config.model)
    
    def _load_dataloader(self):
        from util.dataloader import get_blip3o_dataloader
        from util.dataloader import get_llava_mix665k_dataloader

        self.dataloader_gen = get_blip3o_dataloader(self.config.data.gen, self.accelerator)
        self.dataloader_und = get_llava_mix665k_dataloader(self.config.data.und)
    
    def train(self):
        self.accelerator.print("Training...")


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