import os
import json
import torch
import random


class LLaVAMix665K(torch.utils.data.Dataset):
    def __init__(self, img_path, ann_path):
        self.img_path = img_path
        self.ann_path = ann_path
        self.data = json.load(open(ann_path, "r"))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if "image" in data:
            # load image
            img_path = os.path.join(self.img_path, data["image"])
            # load Q&A pair
            num_qa_pair = len(data["conversations"]) // 2
            qa_index = random.randint(0, num_qa_pair - 1)
            assert data["conversations"][2*qa_index]["from"] == "human"
            assert data["conversations"][2*qa_index+1]["from"] == "gpt"
            question = data["conversations"][2*qa_index]["value"]
            answer = data["conversations"][2*qa_index+1]["value"]

            if "<image>\n" in question:
                question = question.replace("<image>\n", "")
            elif "\n<image>" in question:
                question = question.replace("\n<image>", "")

            item = {
                "question": question,
                "answer": answer,
                "image": img_path,
            }
            return item
        else:
            item = {
                "question": None,
                "answer": None,
            }
            return item

def get_llava_mix665k_dataloader():
    img_path = None
    ann_path = None

    def collate_fn(batch):
        for item in batch:
            if "image" not in item:
                continue

            image_path = item["image"]
            question = item["question"]
            answer = item["answer"]
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": img_path},
                        {"type": "text", "text": question},
                    ],
                }
            ]


        return batch

    dataloader = torch.utils.data.DataLoader(
        LLaVAMix665K(img_path, ann_path),
        batch_size  = 2,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
        drop_last   = True,
        collate_fn  = collate_fn,
    )

    return dataloader