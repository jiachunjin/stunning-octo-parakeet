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
            if pixel_value is None:
                continue
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

        if len(pixel_values) == 0:
            return None
        else:
            pixel_values = torch.stack(pixel_values).squeeze(0)
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

def get_blip3o_dataloader(config, accelerator):
    import glob
    import math
    import webdataset as wds
    import torchvision.transforms as pth_transforms
    from model.internvl.conversation import get_conv_template

    tokenizer = AutoTokenizer.from_pretrained("/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3_5-1B", trust_remote_code=True, use_fast=False)
    IMG_START_TOKEN = "<img>"

    urls = []
    for path in config.wds_path:
        urls.extend(glob.glob(os.path.join(path, "*.tar")))
    accelerator.print(f"Found tar files: {len(urls)}")

    preprocess_gen = pth_transforms.Compose([
        pth_transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        pth_transforms.Resize(config.img_size, max_size=None),
        pth_transforms.CenterCrop(config.img_size),
        pth_transforms.ToTensor(),
        # pth_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    def preprocess_image(image):
        width, height = image.size
        max_size = max(width, height)
        if max_size < config.img_size * 0.75:
            return None
        pixel_values = preprocess_gen(image)

        return pixel_values
    
    def preprocess_text(text):
        if random.random() < config.cfg_drop_rate:
            text = ""

        template = get_conv_template("internvl2_5")
        prompt = f"Generate an image: {text}"

        template.append_message(template.roles[0], prompt)
        template.append_message(template.roles[1], None)
        prompt = template.get_prompt() + IMG_START_TOKEN

        tokenizer_output = tokenizer(
            prompt,
            return_tensors = "pt",
            padding        = "max_length",
            padding_side   = "left",
            truncation     = True,
            max_length     = config.max_seq_length - config.num_img_token,
        )
        input_ids = tokenizer_output["input_ids"]
        attention_mask = tokenizer_output["attention_mask"]

        return input_ids, attention_mask
    
    def collation_fn(batch):
        pixel_values = []
        input_ids_list = []
        attention_mask_list = []

        for sample in batch:
            pixel_value, (input_ids, attention_mask) = sample
            if pixel_value == None:
                continue
            else:
                pixel_values.append(pixel_value)
                input_ids_list.append(input_ids[0])
                attention_mask_list.append(attention_mask[0])

        pixel_values = torch.stack(pixel_values)
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    pipeline = [
        wds.ResampledShards(urls),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.shuffle(bufsize=config.buffer_size, initial=config.buffer_size),
        wds.decode("pil", handler=wds.ignore_and_continue),
        # wds.to_tuple("jpg", "cls"),
        wds.to_tuple("jpg", "txt"),
        wds.map_tuple(preprocess_image, preprocess_text),
        wds.batched(config.batch_size, partial=False, collation_fn=collation_fn),
    ]

    # num_train_examples = 1281167 + 50000
    num_train_examples = 35000000
    global_batch_size = config.batch_size * accelerator.num_processes
    num_workers_per_gpu = config.num_workers

    num_worker_batches = math.ceil(num_train_examples / 
        (global_batch_size * num_workers_per_gpu))
    
    accelerator.print(f"num_worker_batches: {num_worker_batches}")

    train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        train_dataset,
        batch_size  = None,
        num_workers = config.num_workers,
        pin_memory  = True,
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