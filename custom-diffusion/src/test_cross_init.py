import argparse
import torch
import numpy as np
import json
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

from transformers import CLIPTokenizer,CLIPTextModel
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from src.utils import *


@torch.no_grad()
def infer(
    prompt:str,
    n_images:int,
    pretrained_model_name_or_path,
    learned_embed_name_or_path,
    num_inference_steps=50,
    generator=None,
    device='cpu',
):
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer",torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder",torch_dtype=torch.float16).to(device)

    embeds_dict=torch.load(learned_embed_name_or_path)
    tokens=list(embeds_dict.keys())
    embeds = [embeds_dict[token]for token in tokens]

    tokenizer.add_tokens(tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    for i,token_id in enumerate(token_ids):
        text_encoder.get_input_embeddings().weight.data[token_id] = embeds[i]

    prompt=prompt.format(" ".join(tokens))

    pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    input_prompt = [prompt]*n_images
    images = pipe(
        input_prompt,
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=6,
        eta=1
    ).images
    return images, prompt


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a testing script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        required=False,
        default=50,
    )
    parser.add_argument(
        "--num_images_per_prompt",
        required=False,
        default=16,
        type=int,
    )
    parser.add_argument(
        "--save_dir",
        required=False,
        default=None,
        type=str,
    )
    parser.add_argument(
        "--device",
        required=False,
        default="cuda:0",
        type=str,
    )
    parser.add_argument(
        "--learned_embedding_path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--prompt",
        required=False,
        type=str,
        default=None
    )
    parser.add_argument(
        "--n_iter",
        required=False,
        type=int,
        default=1
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    if args.prompt is not None and args.prompt_file is not None:
        raise ValueError('`--prompt` cannot be used with `--prompt_file`')
    
    if args.prompt is not None:
        args.prompt=[args.prompt]

    if args.prompt_file is not None:
        with open(args.prompt_file,'r') as f:
            args.prompt=f.read().splitlines()

    if args.save_dir is not None:
        Path(args.save_dir).mkdir(parents=True,exist_ok=True)

    return args




if __name__ == '__main__':
    args=parse_args()

    generator=None if args.seed is None else torch.Generator(args.device).manual_seed(args.seed)
    save_dir=Path(args.save_dir)
    sample_prompts = {}

    image_index=0
    for prompt in tqdm(args.prompt):
        
        for j in range(args.n_iter):
            
            images, input_prompt =infer(
                prompt=prompt,
                n_images=args.num_images_per_prompt,
                pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                learned_embed_name_or_path=args.learned_embedding_path,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                device=args.device
            )
            # print(type(images[0]))
            #single instances of images in sample folder
            #grid in output_dir
            prompt_grid_name = f"{input_prompt.replace(' ','_')}.png"
            grid_image = np.hstack([np.array(x) for x in images])
            grid_image = Image.fromarray(grid_image)
            grid_image.save(save_dir/prompt_grid_name)

            image_save_path=save_dir.joinpath('samples')
            image_save_path.mkdir(exist_ok=True,parents=True)

            for i,image in enumerate(images):
                stem_name = f'{i+j*args.num_images_per_prompt}'
                sample_prompts[image_index] = input_prompt
                image.save(image_save_path/f'{image_index}.jpg')
                image_index+=1

    with (save_dir / "samples" / "prompts.json").open('w') as f:
        json.dump(sample_prompts, f)