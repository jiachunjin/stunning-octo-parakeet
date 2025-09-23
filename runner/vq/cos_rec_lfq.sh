#!/bin/bash
accelerate launch \
    --config_file config/accelerate_config/deepspeed \
    runner/vq/cos_rec_lfq.py \
    --config config/vq/cos_rec_lfq.yaml