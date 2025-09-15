import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    from transformers import AutoTokenizer
    from util.internvl_preprocess import load_image
    from model.internvl.conversation import get_conv_template

    tokenizer = AutoTokenizer.from_pretrained("/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3_5-1B", trust_remote_code=True, use_fast=False)
    IMG_START_TOKEN = "<img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    IMG_END_TOKEN = "</img>"

    num_image_token = 256

    # img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    img_path = "/data/phd/jinjiachun/dataset/llava_mix665k"
    ann_path = "/data/phd/jinjiachun/dataset/liuhaotian/LLaVA-Instruct-150K/llava_v1_5_mix665k.json"

    def collate_fn(batch):
        pixel_values = []
        questions = []
        answers = []

        for item in batch:
            if "image" not in item:
                continue

            image_path = item["image"]
            question = item["question"]
            answer = item["answer"]

            pixel_value = load_image(image_path, max_num=12)
            question = "<image>\n" + question

            template = get_conv_template("internvl2_5")
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            num_patches_list = [pixel_value.shape[0]] if pixel_value is not None else []

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches_list[0] + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)

            question_inputs = tokenizer(query, return_tensors="pt")
            answer_inputs = tokenizer(answer, return_tensors="pt")

            pixel_values.append(pixel_value)
            questions.append(question_inputs["input_ids"][0])
            answers.append(answer_inputs["input_ids"][0])

        # 移动到循环外面进行stack操作
        pixel_values = torch.stack(pixel_values)
        questions = torch.stack(questions)
        answers = torch.stack(answers)

        return {
            "pixel_values": pixel_values,
            "question": questions,
            "answer": answers,
        }


    dataloader = torch.utils.data.DataLoader(
        LLaVAMix665K(img_path, ann_path),
        batch_size  = 1,
        shuffle     = True,
        num_workers = 1,
        pin_memory  = True,
        drop_last   = True,
        collate_fn  = collate_fn,
    )

    return dataloader

if __name__ == "__main__":
    from transformers import AutoTokenizer
    dataloader = get_llava_mix665k_dataloader()
    tokenizer = AutoTokenizer.from_pretrained("/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3_5-1B", trust_remote_code=True, use_fast=False)

    for batch in dataloader:
        print(batch["pixel_values"].shape, batch["question"].shape, batch["answer"].shape)
        print(tokenizer.decode(batch["question"][0]))
        print(tokenizer.decode(batch["answer"][0]))
        break
        # print(batch)
        # break