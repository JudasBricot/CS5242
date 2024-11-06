#!/bin/bash

# change the data path to actual training data
datasets=(
  "/home/lianxiang/test/cs5242/benchmark_dataset/person_1",
  "/home/lianxiang/test/cs5242/benchmark_dataset/person_2",
  "/home/lianxiang/test/cs5242/benchmark_dataset/person_3"
)


output_dirs=(
  "textual_inversion_output/person_1",
  "textual_inversion_output/person_2",
  "textual_inversion_output/person_3"
)

for i in ${!datasets[@]}; do
  placeholder_token=$(basename "${datasets[i]}")
  
  accelerate launch train_textual_inversion.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --train_data_dir="${datasets[i]}" \
    --placeholder_token="${placeholder_token}" \
    --initializer_token="object" \
    --output_dir="${output_dirs[i]}" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-04 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=2000 &
done

wait

echo "All training completed!"
