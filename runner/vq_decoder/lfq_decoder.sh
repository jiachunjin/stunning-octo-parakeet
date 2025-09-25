#!/bin/bash
accelerate launch \
    --config_file config/accelerate_config/deepspeed \
    runner/vq_decoder/lfq_decoder.py \
    --config config/vq_decoder/lfq_decoder.yaml