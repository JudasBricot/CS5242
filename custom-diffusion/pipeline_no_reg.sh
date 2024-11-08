#!/bin/bash

export MODEL_NAME="/workspace/models/stable-diffusion-v1-5"
export CLASS_DATA_DIR="/workspace/data/diverse_gen/person"
export CLASS_PROMPT="person"

project_type="no_reg"

input_dirs=("/workspace/data/benchmark_dataset/person_1" \
            "/workspace/data/benchmark_dataset/person_2" \
            "/workspace/data/benchmark_dataset/person_3")

output_dirs=("/workspace/output/person_1_${project_type}" \
            "/workspace/output/person_2_${project_type}" \
            "/workspace/output/person_3_${project_type}")

for i in "${!input_dirs[@]}"; do
    input_dir="${input_dirs[i]}"
    output_dir="${output_dirs[i]}"
    echo ${input_dir} ${output_dir}
            #   --learning_rate=5e-6  \
            #--scale_lr 
    accelerate launch src/diffusers_training.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=${input_dir} \
          --class_data_dir=$CLASS_DATA_DIR \
          --output_dir=${output_dir} \
          --instance_prompt="photo of a <new1> ${CLASS_PROMPT}"  \
          --class_prompt=$CLASS_PROMPT \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=5e-6  \
          --lr_warmup_steps=0 \
          --max_train_steps=750 \
          --num_class_images=200 \
          --scale_lr --hflip  \
          --modifier_token "<new1>" \
          --freeze_model "crossattn"
    # Your subprocess command here
done

#run sampling
python batch_evaluate.py \
    --config_path /workspace/custom_diffusion/custom-diffusion/batch_eval_output_no_reg.json \
    --freeze_model "crossattn"

for i in "${!input_dirs[@]}"; do
    sample_dir="${output_dirs[i]}"
    target_dir="${input_dirs[i]}"
    # echo ${sample_dir} ${target_dir} ${sample_dir}/evaluate.pkl
    python customconcept101/evaluate.py --sample_root ${sample_dir} --target_path ${target_dir} --numgen 100 --outpkl ${sample_dir}/evaluate.pkl
done