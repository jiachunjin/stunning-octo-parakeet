#!/bin/bash
accelerate launch \
    --config_file config/accelerate_config/deepspeed \
    runner/vq/llava_vq.py \
    --config config/vq/llava_vq.yaml