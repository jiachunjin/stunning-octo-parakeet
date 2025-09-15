accelerate launch \
--config_file config/accelerate_config/deepspeed \
--main_process_port 30002 \
--num_processes 8 \
runner/down_proj/llava_distill.py \
--config config/down_proj/llava_distill.yaml