#!/bin/bash
export MODEL_NAME="/workspace/models/stable-diffusion-v1-5"
# export INSTANCE_DIR="/workspace/data/sample_data/cat"
# export OUTPUT_DIR="/workspace/output/cat"
# export CLASS_DATA_DIR="/workspace/data/real_reg/samples_cat"

export INSTANCE_DIR="/workspace/data/benchmark_dataset/person_3"
export OUTPUT_DIR="/workspace/output/person_3_1"
export CLASS_DATA_DIR="/workspace/data/real_reg/person"
export CLASS_PROMPT="person"





accelerate launch src/diffusers_training.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=$INSTANCE_DIR \
          --class_data_dir=$CLASS_DATA_DIR \
          --output_dir=$OUTPUT_DIR \
          --with_prior_preservation --prior_loss_weight=1.0 \
          --instance_prompt="photo of a <new1> ${CLASS_PROMPT}"  \
          --class_prompt=$CLASS_PROMPT \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=1000 \
          --num_class_images=200 \
          --scale_lr --hflip  \
          --modifier_token "<new1>"

python src/diffusers_sample.py \
        --delta_ckpt="${OUTPUT_DIR}/delta.bin" \
        --ckpt "/workspace/models/stable-diffusion-v1-5" \
        --prompt "<new1> person on top of a mountain"

export INSTANCE_DIR="/workspace/data/benchmark_dataset/person_1"
export OUTPUT_DIR="/workspace/output/person_1"
export CLASS_DATA_DIR="/workspace/data/real_reg/person"
export CLASS_PROMPT="person"

accelerate launch src/diffusers_training.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=$INSTANCE_DIR \
          --class_data_dir=$CLASS_DATA_DIR \
          --output_dir=$OUTPUT_DIR \
          --with_prior_preservation --prior_loss_weight=1.0 \
          --instance_prompt="photo of a <new1> ${CLASS_PROMPT}"  \
          --class_prompt=$CLASS_PROMPT \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=500 \
          --num_class_images=200 \
          --scale_lr --hflip  \
          --modifier_token "<new1>"

python src/diffusers_sample.py \
        --delta_ckpt="${OUTPUT_DIR}/delta.bin" \
        --ckpt "/workspace/models/stable-diffusion-v1-5" \
        --prompt "<new1> person on top of a mountain"

export INSTANCE_DIR="/workspace/data/benchmark_dataset/person_1"
export OUTPUT_DIR="/workspace/output/person_1_1"
export CLASS_DATA_DIR="/workspace/data/real_reg/person"
export CLASS_PROMPT="person"

accelerate launch src/diffusers_training.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=$INSTANCE_DIR \
          --class_data_dir=$CLASS_DATA_DIR \
          --output_dir=$OUTPUT_DIR \
          --with_prior_preservation --prior_loss_weight=1.0 \
          --instance_prompt="photo of a <new1> ${CLASS_PROMPT}"  \
          --class_prompt=$CLASS_PROMPT \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=1000 \
          --num_class_images=200 \
          --scale_lr --hflip  \
          --modifier_token "<new1>"

python src/diffusers_sample.py \
        --delta_ckpt="${OUTPUT_DIR}/delta.bin" \
        --ckpt "/workspace/models/stable-diffusion-v1-5" \
        --prompt "<new1> person on top of a mountain"

export INSTANCE_DIR="/workspace/data/benchmark_dataset/person_2"
export OUTPUT_DIR="/workspace/output/person_2"
export CLASS_DATA_DIR="/workspace/data/real_reg/person"
export CLASS_PROMPT="person"

accelerate launch src/diffusers_training.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=$INSTANCE_DIR \
          --class_data_dir=$CLASS_DATA_DIR \
          --output_dir=$OUTPUT_DIR \
          --with_prior_preservation --prior_loss_weight=1.0 \
          --instance_prompt="photo of a <new1> ${CLASS_PROMPT}"  \
          --class_prompt=$CLASS_PROMPT \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=5e-6  \
          --lr_warmup_steps=0 \
          --max_train_steps=750 \
          --num_class_images=200 \
          --scale_lr --hflip  \
          --modifier_token "<new1>"

python src/diffusers_sample.py \
        --delta_ckpt="${OUTPUT_DIR}/delta.bin" \
        --ckpt "/workspace/models/stable-diffusion-v1-5" \
        --prompt "<new1> person on top of a mountain"

export INSTANCE_DIR="/workspace/data/benchmark_dataset/person_2"
export OUTPUT_DIR="/workspace/output/person_2_1"
export CLASS_DATA_DIR="/workspace/data/real_reg/person"
export CLASS_PROMPT="person"

accelerate launch src/diffusers_training.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=$INSTANCE_DIR \
          --class_data_dir=$CLASS_DATA_DIR \
          --output_dir=$OUTPUT_DIR \
          --with_prior_preservation --prior_loss_weight=1.0 \
          --instance_prompt="photo of a <new1> ${CLASS_PROMPT}"  \
          --class_prompt=$CLASS_PROMPT \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=1000 \
          --num_class_images=200 \
          --scale_lr --hflip  \
          --modifier_token "<new1>"

python src/diffusers_sample.py \
        --delta_ckpt="${OUTPUT_DIR}/delta.bin" \
        --ckpt "/workspace/models/stable-diffusion-v1-5" \
        --prompt "<new1> person on top of a mountain"