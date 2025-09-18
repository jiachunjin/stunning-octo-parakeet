#!/bin/bash

# 运行优化后的联合训练脚本

# 使用accelerate运行优化版本
accelerate launch \
    --config_file config/accelerate_config/deepspeed \
    runner/down_proj/joint_training_optimized.py \
    --config config/down_proj/joint_training.yaml
