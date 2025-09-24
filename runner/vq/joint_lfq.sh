#!/bin/bash
accelerate launch \
    --config_file config/accelerate_config/deepspeed \
    runner/vq/joint_lfq.py \
    --config config/vq/joint_lfq.yaml