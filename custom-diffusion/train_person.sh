#!/bin/bash
export MODEL_NAME="/workspace/models/stable-diffusion-v1-5"
export CLASS_DATA_DIR="/workspace/data/real_reg/person"
export CLASS_PROMPT="person"

# input_dirs=("/workspace/data/benchmark_dataset/person_1" \
#             "/workspace/data/benchmark_dataset/person_2" \
#             "/workspace/data/benchmark_dataset/person_3")
# output_dirs=("/workspace/output/person_1" \
#             "/workspace/output/person_2" \
#             "/workspace/output/person_3")

input_dirs=("/workspace/data/benchmark_dataset/person_1")
output_dirs=("/workspace/output/person_1")

for i in "${!input_dirs[@]}"; do
    input_dir="${input_dirs[i]}"
    output_dir="${output_dirs[i]}"
    # echo ${input_dir} ${output_dir}
    accelerate launch src/diffusers_training.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=${input_dir} \
          --class_data_dir=$CLASS_DATA_DIR \
          --output_dir=${output_dir} \
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
    # Your subprocess command here
done