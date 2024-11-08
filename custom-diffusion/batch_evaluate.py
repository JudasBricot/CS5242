import json
import os
# from inference_custom_diff import inference, parse_config, InferenceConfig
from tqdm import tqdm 
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from diffusers import DPMSolverMultistepScheduler
from src.diffusers_model_pipeline import CustomDiffusionPipeline, CustomDiffusionXLPipeline
import argparse

def sample(ckpt, delta_ckpt, prompts, batch_size, freeze_model, compress, is_sdxl=False):
    model_id = ckpt
    if is_sdxl:
        pipe = CustomDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None).to("cuda")
    else:
        pipe = CustomDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None).to("cuda")
    pipe.load_model(delta_ckpt, compress)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    outdir = os.path.dirname(delta_ckpt)
    generator = torch.Generator(device='cuda').manual_seed(42)



    all_images = []
    all_prompts = []
    for prompt in tqdm(prompts):
        input_prompt = [prompt]*batch_size
        images = pipe(input_prompt, num_inference_steps=200, guidance_scale=6., eta=1., generator=generator).images
        all_images += images
        all_prompts+= input_prompt
        images = np.hstack([np.array(x) for x in images])
        images = Image.fromarray(images)
        name = '-'.join(prompt[:50].split())
        images.save(f'{outdir}/{name}.png')

    sample_prompt = {}
    os.makedirs(f'{outdir}/samples', exist_ok=True)
    for i, (im, im_prompt) in enumerate(zip(all_images, all_prompts)):
        sample_prompt[i] = im_prompt
        im.save(f'{outdir}/samples/{i}.jpg')

    with open(f'{outdir}/samples/prompts.json', 'w') as f:
        json.dump(sample_prompt, f)

def start_batch_inference(config_path, 
                        is_multi_concept=False, batch_size=5,
                        freeze_model='crossattn_kv', compress=False,
                        is_sdxl=False,
                        uses_cr_embedding=False):
    
    with open(config_path) as f:
        config_dict =  json.load(f)

    if is_multi_concept:
        raise NotImplementedError("No multi concept yet!")
    else:

        sample_instances = []
        for config in config_dict:
            sample_instances.append(
                (
                    config['base-model'], 
                    config['delta-checkpoint'], 
                    config['token'], 
                    config['class-instance'],
                    config['prompt-file']
                )
            )
        
        for sd_path, delta_chkpt, token, class_instance, prompt_file in tqdm(sample_instances):
            print(f"\t sd_path: {sd_path}")
            print(f"\t delta_ckpt: {delta_chkpt}")
            print(f"\t from_file: {prompt_file}")
            
            #need to replace the prompt
            #sanity check
            os.path.exists(sd_path)
            os.path.exists(delta_chkpt)
            prompts = []

            if uses_cr_embedding:
                #uses cross initialization text embedding
                #have to extract out the tokens
                embeds_dict=torch.load(delta_chkpt)
                modifier_tokens = list(embeds_dict['modifier_token'].keys())
                modifier_tokens = " ".join(modifier_tokens)
                prompt_token = f"{modifier_tokens} {class_instance}"
            else:
                prompt_token = f"{token} {class_instance}"
            print("Prompt token:", prompt_token)

            with open(prompt_file) as f:
                for line in f:
                    line = line.strip().format(prompt_token)
                    prompts.append(line)
            
            sample(sd_path, delta_chkpt, prompts, 
                    batch_size=batch_size,freeze_model=freeze_model, compress=compress, is_sdxl=is_sdxl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None, required=True)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--is_multi_concept', action="store_true", default=False)
    parser.add_argument('--freeze_model', type=str, default="crossattn_kv")
    parser.add_argument('--compress', action="store_true", default=False)
    parser.add_argument('--is_sdxl', action="store_true", default=False)
    parser.add_argument('--uses_cr_embedding', action="store_true", default=False)

    args = parser.parse_args()
    # json_path = "/workspace/custom_diffusion/custom-diffusion/customconcept101/dataset_person_only.json"
    # config_checkpoint = "/workspace/custom_diffusion/custom-diffusion/batch_eval_output_dir.json"
    # json_path = "/workspace/custom_diffusion/custom-diffusion/customconcept101/dataset_cats_only.json"
    # config_checkpoint = "/workspace/custom_diffusion/custom-diffusion/batch_eval_output_dir_cat.json"
    start_batch_inference(config_path=args.config_path, 
                        is_multi_concept=args.is_multi_concept, 
                        batch_size=args.batch_size,
                        freeze_model=args.freeze_model, 
                        compress=args.compress,
                        is_sdxl=args.is_sdxl, 
                        uses_cr_embedding=args.uses_cr_embedding)