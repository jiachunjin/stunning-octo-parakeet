import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# model_checkpoint = "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3_5-1B-Pretrained-HF"
# processor = AutoProcessor.from_pretrained("/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3_5-1B-HF")
# model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map="auto", dtype=torch.bfloat16)

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
#             {"type": "text", "text": "这张图怎么样？"},
#         ],
#     }
# ]

# # messages = ["what is the weather in beijing?"]
# # tokenizer = processor.tokenizer
# # inputs = tokenizer(messages, return_tensors="pt")
# inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
# # print(processor.decode(inputs.input_ids[0]))
# print(inputs.input_ids.shape)


# generate_ids = model.generate(**inputs, max_new_tokens=50)
# decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

# print(decoded_output)

# ---------------------------------------------
def test_mme():
    from datasets import load_dataset
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_checkpoint = "/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3_5-1B-Pretrained-HF"
    model_name = model_checkpoint.split("/")[-1]
    processor = AutoProcessor.from_pretrained("/data/phd/jinjiachun/ckpt/OpenGVLab/InternVL3_5-1B-HF")
    model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map="cuda", dtype=torch.bfloat16)

    data_files = {
        "test": "/data/phd/jinjiachun/dataset/benchmark/darkyarding/MME/data/test-*-of-*.parquet"
    }
    dataset = load_dataset("parquet", data_files=data_files)

    for i, data in enumerate(dataset["test"]):
        img_name = data["question_id"].split("/")[-1]
        category = data["category"]
        image = data["image"].convert("RGB")
        question = data["question"]
        gt_answer = data["answer"]

        messages = [
            {
                "role": "user",
                "content": [
                    # {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

        generate_ids = model.generate(**inputs, max_new_tokens=50)
        answer = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        os.makedirs("evaluation/understanding/mme/model_name", exist_ok=True)
        with open(f"evaluation/understanding/mme/model_name/{category}.txt", "a") as f:
            line = f"{img_name}\t{question}\t{gt_answer}\t{answer}\n"
            f.write(line)

if __name__ == "__main__":
    test_mme()