import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from model.internvl.modeling_internvl_chat import InternVLChatModel


# model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
model_checkpoint = "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3_5-1B"
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = InternVLChatModel.from_pretrained(model_checkpoint).to(torch.bfloat16)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "Please describe the image explicitly."},
        ],
    }
]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

generate_ids = model.generate(**inputs, max_new_tokens=50)
decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

print(decoded_output)