import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import clip
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
import timm
import argparse

# for clip
def preprocess_image_clip(image_path, size=224):
    preprocess = Compose([
        Resize(size, interpolation=Image.BICUBIC),
        CenterCrop(size),
        lambda img: img.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])
    image = Image.open(image_path)
    return preprocess(image)

# for dino
def preprocess_image_dino(image_path, size=224):
    preprocess = Compose([
        Resize(size, interpolation=Image.BICUBIC),
        CenterCrop(size),
        lambda img: img.convert("RGB"),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)),
    ])
    image = Image.open(image_path)
    return preprocess(image)

def load_dino_model(device):
    dino_model = timm.create_model('vit_small_patch16_224_dino', pretrained=True)
    dino_model.to(device)
    dino_model.eval()
    return dino_model

class PromptProcessor:
    def __init__(self, append=False, prefix='A photo depicts'):
        self.append = append
        self.prefix = prefix if append else ''

        if self.append and self.prefix and self.prefix[-1] != ' ':
            self.prefix += ' '

    def process(self, prompts):
        processed_prompts = [self.prefix + prompt for prompt in prompts]
        tokenized_prompts = clip.tokenize(processed_prompts, truncate=True)
        return tokenized_prompts

# CLIP score
def calculate_clip_score(model, device, prompt_processor, prompts, image_groups):
    scores = []
    for prompt, images in zip(prompts, image_groups):
        images_tensor = [preprocess_image_clip(img).unsqueeze(0) for img in images]
        images_tensor = torch.cat(images_tensor, dim=0).to(device)
        tokenized_prompt = prompt_processor.process([prompt]).to(device)

        with torch.no_grad():
            if isinstance(model, nn.DataParallel):
                image_features = model.module.encode_image(images_tensor)
                text_features = model.module.encode_text(tokenized_prompt)
            else:
                image_features = model.encode_image(images_tensor)
                text_features = model.encode_text(tokenized_prompt)
            image_features = image_features.cpu().numpy()
            text_features = text_features.cpu().numpy()
        image_features /= np.linalg.norm(image_features, axis=1, keepdims=True)
        text_features /= np.linalg.norm(text_features, axis=1, keepdims=True)

        # cosine similarity
        clip_scores = image_features @ text_features.T  # (num_images, 1)
        mean_score = np.mean(clip_scores).astype(float)
        var_score = np.var(clip_scores).astype(float)

        scores.append((mean_score, var_score))

    return scores

# DINO
def calculate_dino_features(model, device, image_paths):
    batch_size = 32 
    features = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = [preprocess_image_dino(img).unsqueeze(0) for img in batch_paths]
            batch_images = torch.cat(batch_images, dim=0).to(device)
            if isinstance(model, nn.DataParallel):
                feature = model.module.forward_features(batch_images)
            else:
                feature = model.forward_features(batch_images)
            feature = torch.nn.functional.adaptive_avg_pool2d(feature, (1, 1))
            feature = feature.view(feature.size(0), -1)
            feature = feature.cpu().numpy()
            features.append(feature)
    features = np.vstack(features)
    return features

# Main Evaluation Process
def evaluate_generated_images(root_dir, prompt_file, device, num_gpus, append_prefix=False, prefix='A photo depicts'):
    model_clip, _ = clip.load("ViT-B/32", device=device)
    if num_gpus > 1:
        model_clip = nn.DataParallel(model_clip)
    model_clip.eval()
    
    model_dino = load_dino_model(device)
    if num_gpus > 1:
        model_dino = nn.DataParallel(model_dino)
    model_dino.eval()
    
    prompt_processor = PromptProcessor(append=append_prefix, prefix=prefix)
    
    with open(prompt_file, 'r') as f:
        prompts = [line.strip().replace("{}", "person") for line in f.readlines()]
    
    results = {}
    
    for subdir in tqdm(os.listdir(root_dir), desc="Processing instances"):
        subdir_path = os.path.join(root_dir, subdir)
        samples_dir = os.path.join(subdir_path, 'samples')
        
        if not os.path.isdir(samples_dir) or len([f for f in os.listdir(samples_dir) if f.lower().endswith('.jpg')]) < 100:
            continue
        image_groups = []
        for j in range(0, 100, 5):
            group = [os.path.join(samples_dir, f"{i}.jpg") for i in range(j, j+5)]
            if all(os.path.exists(img) for img in group):
                image_groups.append(group)
            else:
                break
        if len(image_groups) * 5 != 100:
            continue
        
        clip_scores = calculate_clip_score(model_clip, device, prompt_processor, prompts, image_groups)
        dino_features = []
        for group in image_groups:
            features = calculate_dino_features(model_dino, device, group)
            mean_feature = np.mean(features, axis=0)
            dino_features.append(mean_feature)
        dino_features = np.vstack(dino_features)

        instance_id = subdir.split('_')[0]
        results[instance_id] = {
            "overall_clip_score": np.mean([score[0] for score in clip_scores]).astype(float),
            "per_prompt_scores": {
                prompt: {"clip_score": score[0], "variance": score[1]} for prompt, score in zip(prompts, clip_scores)
            },
            "dino_mean_features": dino_features.tolist()
        }

    if results:
        overall_clip_score = np.mean([res["overall_clip_score"] for res in results.values()])
        print(f"Overall CLIPScore: {overall_clip_score:.4f}")
    else:
        print("No valid instances found.")

    results_path = Path(root_dir) / "clip_dino_evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated images using CLIP and DINO.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing generated images.")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the prompt file.")
    parser.add_argument("--append_prefix", action='store_true', help="Whether to append a prefix to prompts.")
    parser.add_argument("--prefix", type=str, default='A photo depicts', help="Prefix to append to prompts.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    evaluate_generated_images(
        root_dir=args.root_dir,
        prompt_file=args.prompt_file,
        device=device,
        num_gpus=num_gpus,
        append_prefix=args.append_prefix,
        prefix=args.prefix
    )
