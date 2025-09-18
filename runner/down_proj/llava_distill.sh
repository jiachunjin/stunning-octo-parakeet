#!/bin/bash
accelerate launch \
    --config_file config/accelerate_config/deepspeed \
    runner/down_proj/llava_distill.py \
    --config config/down_proj/llava_distill.yaml