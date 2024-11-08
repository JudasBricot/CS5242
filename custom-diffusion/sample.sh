#!/bin/bash
export MODEL_NAME="/workspace/models/stable-diffusion-v1-5"
# export INSTANCE_DIR="/workspace/data/sample_data/cat"
# export OUTPUT_DIR="/workspace/output/cat"
# export CLASS_DATA_DIR="/workspace/data/real_reg/samples_cat"

export INSTANCE_DIR="/workspace/data/benchmark_dataset/person_3"
export OUTPUT_DIR="/workspace/output/person_3_1"
export CLASS_DATA_DIR="/workspace/data/real_reg/person"
export CLASS_PROMPT="person"

python src/diffusers_sample.py \
        --delta_ckpt="${OUTPUT_DIR}/delta.bin" \
        --ckpt "/workspace/models/stable-diffusion-v1-5" \
        --prompt "<new1> person on top of a mountain"