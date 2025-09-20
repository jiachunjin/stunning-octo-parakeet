import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch
import random
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque

class AsyncDataLoader:
    """异步数据加载器，预加载数据以减少GPU等待时间"""
    
    def __init__(self, dataloader, buffer_size=4, num_prefetch_workers=2):
        self.dataloader = dataloader
        self.buffer_size = buffer_size
        self.num_prefetch_workers = num_prefetch_workers
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.prefetch_threads = []
        self._start_prefetch()
    
    def _start_prefetch(self):
        """启动预取线程"""
        for _ in range(self.num_prefetch_workers):
            thread = threading.Thread(target=self._prefetch_worker)
            thread.daemon = True
            thread.start()
            self.prefetch_threads.append(thread)
    
    def _prefetch_worker(self):
        """预取工作线程"""
        try:
            for batch in self.dataloader:
                if self.stop_event.is_set():
                    break
                try:
                    self.buffer.put(batch, timeout=1)
                except queue.Full:
                    # 如果缓冲区满了，丢弃最老的batch
                    try:
                        self.buffer.get_nowait()
                        self.buffer.put(batch, timeout=1)
                    except queue.Empty:
                        pass
        except Exception as e:
            print(f"Prefetch worker error: {e}")
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.stop_event.is_set():
            raise StopIteration
        
        try:
            batch = self.buffer.get(timeout=5)
            return batch
        except queue.Empty:
            raise StopIteration
    
    def __del__(self):
        self.stop_event.set()
        for thread in self.prefetch_threads:
            thread.join(timeout=1)

class OptimizedLLaVAMix665K(torch.utils.data.Dataset):
    """优化的LLaVA数据集，减少重复计算"""
    
    def __init__(self, img_path, ann_path, tokenizer, max_seq_length=1024):
        self.img_path = img_path
        self.ann_path = ann_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = json.load(open(ann_path, "r"))
        
        # 预计算一些常量
        self.IMG_START_TOKEN = "<img>"
        self.IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        self.IMG_END_TOKEN = "</img>"
        self.num_image_token = 256
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if "image" not in data:
            return None

        # 加载图像
        img_path = os.path.join(self.img_path, data["image"])
        if not os.path.exists(img_path):
            return None
            
        # 加载Q&A对
        num_qa_pair = len(data["conversations"]) // 2
        if num_qa_pair == 0:
            return None
            
        qa_index = random.randint(0, num_qa_pair - 1)
        assert data["conversations"][2*qa_index]["from"] == "human"
        assert data["conversations"][2*qa_index+1]["from"] == "gpt"
        question = data["conversations"][2*qa_index]["value"]
        answer = data["conversations"][2*qa_index+1]["value"]

        if "<image>\n" in question:
            question = question.replace("<image>\n", "")
        elif "\n<image>" in question:
            question = question.replace("\n<image>", "")

        return {
            "question": question,
            "answer": answer,
            "image": img_path,
        }

def get_optimized_llava_dataloader(config):
    """优化的LLaVA数据加载器"""
    from transformers import AutoTokenizer
    from util.internvl_preprocess import load_image
    from model.internvl.conversation import get_conv_template

    tokenizer = AutoTokenizer.from_pretrained(
        "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3_5-1B", 
        trust_remote_code=True, 
        use_fast=False
    )
    
    img_path = "/data/phd/jinjiachun/dataset/llava_mix665k"
    ann_path = "/data/phd/jinjiachun/dataset/liuhaotian/LLaVA-Instruct-150K/llava_v1_5_mix665k.json"

    def collate_fn(batch):
        # 过滤掉None值
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None

        pixel_values = []
        input_ids = []
        attention_mask = []
        answer_mask = []

        for item in batch:
            image_path = item["image"]
            question = item["question"]
            answer = item["answer"]

            # 异步加载图像
            pixel_value = load_image(image_path, max_num=12)
            if pixel_value is None:
                continue
                
            question = "<image>\n" + question

            template = get_conv_template("internvl2_5")
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            num_patches_list = [pixel_value.shape[0]] if pixel_value is not None else []
            image_tokens = self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * self.num_image_token * num_patches_list[0] + self.IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)

            # 批量tokenization
            tokenizer_output = tokenizer(
                query + answer,
                return_tensors="pt",
                padding="max_length",
                padding_side="right",
                truncation=True,
                max_length=config.max_seq_length,
            )

            input_ids_batch = tokenizer_output["input_ids"]
            attention_mask_batch = tokenizer_output["attention_mask"]
            
            # 计算answer mask
            query_tokens = tokenizer(query, return_tensors="pt")
            answer_tokens = tokenizer(answer, return_tensors="pt")
            query_length = query_tokens["input_ids"].shape[1]
            answer_length = answer_tokens["input_ids"].shape[1]
            
            answer_mask_batch = torch.zeros_like(input_ids_batch, dtype=torch.bool)
            if query_length + answer_length <= config.max_seq_length:
                answer_mask_batch[0, query_length:query_length + answer_length] = True
            else:
                actual_length = attention_mask_batch.sum().item()
                answer_start = max(0, actual_length - answer_length)
                answer_mask_batch[0, answer_start:actual_length] = True
            
            pixel_values.append(pixel_value)
            input_ids.append(input_ids_batch[0])
            attention_mask.append(attention_mask_batch[0])
            answer_mask.append(answer_mask_batch[0])

        if len(pixel_values) == 0:
            return None
        else:
            pixel_values = torch.stack(pixel_values).squeeze(1)
            input_ids = torch.stack(input_ids)
            attention_mask = torch.stack(attention_mask)
            answer_mask = torch.stack(answer_mask)

            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "answer_mask": answer_mask,
            }

    # 创建数据集
    dataset = OptimizedLLaVAMix665K(img_path, ann_path, tokenizer, config.max_seq_length)
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=True,  # 保持worker进程活跃
    )
    
    # 包装为异步数据加载器
    return AsyncDataLoader(dataloader, buffer_size=4, num_prefetch_workers=2)

