#!/bin/bash

# change the data path to actual training data
datasets=(
  "benchmark_dataset/person_1"
  "benchmark_dataset/person_2"
  "benchmark_dataset/person_3"
)

datasets_formated_structure=(
  "temp/dataset/person_1"
  "temp/dataset/person_2"
  "temp/dataset/person_3"
)

output_dirs=(
  "output/models/person_1"
  "output/models/person_2"
  "output/models/person_3"
)

for i in ${!datasets[@]}; do
  placeholder_token=$(basename "${datasets[i]}")
  
  python create_formated_dataset.py \
    --image_dir="${datasets[i]}" \
    --prompt="A photo of a <new1> person" \
    --output_dir="${datasets_formated_structure[i]}" &

  accelerate launch train_text_to_image_lora.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --dataset_name="${datasets_formated_structure[i]}" \
    --output_dir="${output_dirs[i]}" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-05 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=2000 \
    --image_column image \
    --caption_column text &
done

wait

echo "All training completed!"
