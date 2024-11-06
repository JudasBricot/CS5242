from diffusers import StableDiffusionPipeline
import os

model_path = "runwayml/stable-diffusion-v1-5"

# Please change the embed_dir to the actual path you just put your trained embeddings in
embed_dir = "/home/lianxiang/test/cs5242/textual_inversion_output"
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(model_path)

# change prompts to align with custom101
prompts = {
    "person_2": [
        "a portrait photo of person_2 in a natural setting",
        "person_2 smiling in a cityscape",
        "a candid shot of person_2 with friends",
        "person_2 in formal attire at an event",
    ]
}

for embed_file in os.listdir(embed_dir):
    if embed_file.endswith(".safetensors"):
        embed_path = os.path.join(embed_dir, embed_file)
        concept_name = os.path.splitext(embed_file)[0]
        
        pipe.load_textual_inversion(embed_path, token=concept_name)
        
        for i, prompt in enumerate(prompts.get(concept_name, [])):
            full_prompt = prompt.replace(concept_name, f"<{concept_name}>")
            image = pipe(full_prompt).images[0]
            image_path = os.path.join(output_dir, f"{concept_name}_{i+1}.png")
            image.save(image_path)
            print(f"Pictures saved to {image_path}")
