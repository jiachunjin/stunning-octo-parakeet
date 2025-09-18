"""
改进的数据加载器，支持batch size > 1的情况
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from util.dataloader import LLaVAMix665K


def get_llava_mix665k_dataloader_batched(batch_size=4, num_workers=4):
    """
    支持批量处理的LLaVA数据加载器
    
    Args:
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
    """
    from transformers import AutoTokenizer
    from util.internvl_preprocess import load_image
    from model.internvl.conversation import get_conv_template

    tokenizer = AutoTokenizer.from_pretrained(
        "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3_5-1B", 
        trust_remote_code=True, 
        use_fast=False
    )
    
    IMG_START_TOKEN = "<img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    IMG_END_TOKEN = "</img>"
    
    num_image_token = 256
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    img_path = "/data/phd/jinjiachun/dataset/llava_mix665k"
    ann_path = "/data/phd/jinjiachun/dataset/liuhaotian/LLaVA-Instruct-150K/llava_v1_5_mix665k.json"

    def collate_fn_batched(batch):
        """
        改进的collate函数，支持批量处理和padding
        """
        pixel_values_list = []
        questions_list = []
        answers_list = []
        valid_indices = []

        # 处理每个样本
        for idx, item in enumerate(batch):
            if "image" not in item:
                continue

            image_path = item["image"]
            question = item["question"]
            answer = item["answer"]

            pixel_value = load_image(image_path, max_num=12)
            if pixel_value is None:
                continue
                
            # 构建对话模板
            question = "<image>\n" + question
            template = get_conv_template("internvl2_5")
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            # 处理图像token
            num_patches_list = [pixel_value.shape[0]] if pixel_value is not None else []
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches_list[0] + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)

            # tokenize文本
            question_inputs = tokenizer(query, return_tensors="pt")
            answer_inputs = tokenizer(answer, return_tensors="pt")

            pixel_values_list.append(pixel_value)
            questions_list.append(question_inputs["input_ids"][0])  # 移除batch维度
            answers_list.append(answer_inputs["input_ids"][0])      # 移除batch维度
            valid_indices.append(idx)

        # 如果没有有效样本，返回None
        if len(pixel_values_list) == 0:
            return None

        # 处理图像数据
        # 假设所有图像具有相同的形状，直接stack
        pixel_values = torch.stack(pixel_values_list)  # (B, C, H, W) 或 (B, num_patches, C, H, W)

        # 处理文本数据 - 使用padding
        # pad_sequence默认padding_value=0，我们需要使用tokenizer的pad_token_id
        questions_padded = pad_sequence(questions_list, batch_first=True, padding_value=pad_token_id)
        answers_padded = pad_sequence(answers_list, batch_first=True, padding_value=pad_token_id)

        # 创建attention mask
        # attention mask: 1表示真实token，0表示padding
        question_attention_mask = torch.zeros_like(questions_padded, dtype=torch.bool)
        answer_attention_mask = torch.zeros_like(answers_padded, dtype=torch.bool)
        
        for i, q in enumerate(questions_list):
            question_attention_mask[i, :len(q)] = True
        
        for i, a in enumerate(answers_list):
            answer_attention_mask[i, :len(a)] = True

        return {
            "pixel_values": pixel_values,
            "question": questions_padded,
            "answer": answers_padded,
            "question_attention_mask": question_attention_mask,
            "answer_attention_mask": answer_attention_mask,
        }

    dataloader = torch.utils.data.DataLoader(
        LLaVAMix665K(img_path, ann_path),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_batched,
    )

    return dataloader


def test_batched_dataloader():
    """测试批量数据加载器"""
    from transformers import AutoTokenizer
    
    # 加载tokenizer用于解码测试
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3_5-1B", 
        trust_remote_code=True, 
        use_fast=False
    )
    
    # 创建数据加载器
    dataloader = get_llava_mix665k_dataloader_batched(batch_size=4)
    
    # 测试一个批次
    for batch in dataloader:
        if batch is None:
            continue
            
        print(f"Batch shapes:")
        print(f"  pixel_values: {batch['pixel_values'].shape}")
        print(f"  question: {batch['question'].shape}")
        print(f"  answer: {batch['answer'].shape}")
        print(f"  question_attention_mask: {batch['question_attention_mask'].shape}")
        print(f"  answer_attention_mask: {batch['answer_attention_mask'].shape}")
        
        # 显示第一个样本的文本内容
        print(f"\nFirst sample:")
        print(f"  Question: {tokenizer.decode(batch['question'][0][batch['question_attention_mask'][0]])}")
        print(f"  Answer: {tokenizer.decode(batch['answer'][0][batch['answer_attention_mask'][0]])}")
        
        break


if __name__ == "__main__":
    test_batched_dataloader()
