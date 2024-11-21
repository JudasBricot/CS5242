#!/bin/bash

# Replace with model path or name
model_name_or_path="runwayml/stable-diffusion-v1-5"

# Replace with lora weights dir
lora_weights_dir="output/models/person_1/checkpoints"

# Replace with lora weights name
lora_weights_filename="pytorch_lora_weights.safetensors" 

# Emplacement of outputed images
output_dir="generated_images"

# Replace with prompts, "<new1> person" is the name of the concept learned when using the CustomConcept101 dataset
prompts=(
  "a portrait photo of <new1> person in a natural setting"
  "<new1> person smiling in a cityscape"
  "a candid shot of <new1> person with friends"
  "<new1> person in formal attire at an event"
)

python generate_image_example.py \
  --model_name_or_path="$model_name_or_path" \
  --lora_weights_dir="$lora_weights_dir" \
  --lora_weights_filename="$lora_weights_filename" \
  --output_dir="$output_dir" \
  --prompts=prompts