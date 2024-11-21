import torch
from diffusers import StableDiffusionPipeline
import argparse
import os

# # Replace with model path or name
# model_name_or_path = "runwayml/stable-diffusion-v1-5"

# # Replace with lora weights dir
# lora_weights_dir = "output/person_1/checkpoints"

# # Replace with lora weights name
# lora_weights_name = "pytorch_lora_weights.safetensors" 

# output_dir = "generated_images"

# # Replace with prompts, "<new1> person" is the name of the concept learned when using the CustomConcept101 dataset
# prompts = [
#     "a portrait photo of <new1> person in a natural setting",
#     "<new1> person smiling in a cityscape",
#     "a candid shot of <new1> person with friends",
#     "<new1> person in formal attire at an event"
# ]

def generate_images(model_name_or_path, 
                   lora_weights_dir, 
                   lora_weights_filename, 
                   output_dir,
                   prompts):
  print(prompts)
  device = "cuda" if torch.cuda.is_available() else "cpu"

  pipe = StableDiffusionPipeline.from_pretrained(model_name_or_path)
  pipe.load_lora_weights(lora_weights_dir, weight_name=lora_weights_filename)
  pipe.to(device)

  for i, prompt in enumerate(prompts):
    image = pipe(prompt).images[0]
    image.save(output_dir + str(i) + ".png")


def parse_arg():
  parser = argparse.ArgumentParser(description="Script to generate images with lora weights.")
  parser.add_argument(
    "--model_name_or_path",
    type=str,
    default=None,
    required=True,
    help="Name or path of the model to be used.",
  )

  parser.add_argument(
  "--lora_weights_dir",
  type=str,
  default=None,
  required=True,
  help="Directory where the weights of the lora are stored.",
  )

  parser.add_argument(
  "--lora_weights_filename",
  type=str,
  default=None,
  required=True,
  help="Name of the lora weights file.",
  )

  parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    required=True,
    help="Path to store the generated images",
  )

  parser.add_argument(
    "--prompts",
    type=str,
    default=None,
    required=True,
    help="Prompts to be used for the generation",
  )

  args = parser.parse_args()
  env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
  if env_local_rank != -1 and env_local_rank != args.local_rank:
    args.local_rank = env_local_rank

  return args
  
if __name__ == "__main__":
  args = parse_arg()
  generate_images(args.model_name_or_path, 
                  args.lora_weights_dir, 
                  args.lora_weights_filename, 
                  args.output_dir,
                  args.prompts)
  