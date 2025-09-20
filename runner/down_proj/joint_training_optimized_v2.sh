#!/bin/bash

# 优化的联合训练脚本
# 使用多GPU和优化的配置

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

# 设置环境变量以优化性能
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# 使用accelerate启动多GPU训练
accelerate launch \
    --config_file config/accelerate_config/deepspeed \
    --num_processes 4 \
    --main_process_port 29500 \
    runner/down_proj/joint_training_optimized_v2.py \
    --config config/down_proj/joint_training_optimized.yaml

