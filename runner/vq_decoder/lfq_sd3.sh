#!/bin/bash
accelerate launch \
    --config_file config/accelerate_config/deepspeed \
    runner/vq_decoder/lfq_sd3.py \
    --config config/vq_decoder/sd3_lfq.yaml