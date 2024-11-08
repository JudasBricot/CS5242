#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME="/workspace/models/stable-diffusion-v1-5"

project_type="cross_init_default"

input_dirs=("/workspace/data/benchmark_dataset/person_1" \
            "/workspace/data/benchmark_dataset/person_2" \
            "/workspace/data/benchmark_dataset/person_3")

output_dirs=("/workspace/output/person_1_${project_type}" \
            "/workspace/output/person_2_${project_type}" \
            "/workspace/output/person_3_${project_type}")

for i in "${!input_dirs[@]}"; do
    input_dir="${input_dirs[i]}"
    output_dir="${output_dirs[i]}"
    python train_cross_init.py \
        --save_steps 100 \
        --only_save_embeds \
        --placeholder_token "<new1>" \
        --train_batch_size 8 \
        --scale_lr \
        --n_persudo_tokens 2 \
        --reg_weight "1e-5" \
        --learning_rate 0.000625 \
        --max_train_step 320 \
        --train_data_dir ${input_dir} \
        --celeb_path "./wiki_names_v2.txt" \
        --pretrained_model_name_or_path $MODEL_NAME \
        --output_dir ${output_dir} 
done

for i in "${!input_dirs[@]}"; do
    input_dir="${input_dirs[i]}"
    output_dir="${output_dirs[i]}"
    python test_cross_init.py \
        --pretrained_model_name_or_path $MODEL_NAME \
        --num_inference_steps 200 \
        --learned_embedding_path "${output_dir}/learned_embeds.bin" \
        --prompt_file "/workspace/custom_diffusion/custom-diffusion/customconcept101/prompts/person.txt" \
        --save_dir ${output_dir} \
        --num_images_per_prompt=5 \
        --n_iter=1 \
        --seed=42
done

for i in "${!input_dirs[@]}"; do
    sample_dir="${output_dirs[i]}"
    target_dir="${input_dirs[i]}"
    # echo ${sample_dir} ${target_dir} ${sample_dir}/evaluate.pkl
    python /workspace/custom_diffusion/custom-diffusion/customconcept101/evaluate.py --sample_root ${sample_dir} --target_path ${target_dir} --numgen 100 --outpkl ${sample_dir}/evaluate.pkl
done