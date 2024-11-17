# pip install insightface onnxruntime-gpu tqdm

import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
import torch
import argparse
from concurrent.futures import ThreadPoolExecutor

app_instances = []

def initialize_app_instances(gpu_count):
    global app_instances
    for i in range(gpu_count):
        app = FaceAnalysis(providers=['CUDAExecutionProvider'], provider_options=[{"device_id": i}])
        app.prepare(ctx_id=i, det_size=(640, 640))
        app_instances.append(app)

def extract_embeddings(image_path, app):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return None
    faces = app.get(img)
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
    embedding = faces[0].embedding
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def compute_average_embedding(embeddings):
    if not embeddings:
        return None
    avg_emb = np.mean(embeddings, axis=0)
    avg_emb = avg_emb / np.linalg.norm(avg_emb)
    return avg_emb

def load_ground_truth_embeddings(gt_root, generated_ids):
    gt_embeddings = {}
    print("Loading Ground Truth embeddings...")
    for instance_id in tqdm(generated_ids, desc="Loading GT embeddings"):
        instance_path = os.path.join(gt_root, instance_id)
        if not os.path.isdir(instance_path):
            print(f"Ground Truth instance '{instance_id}' does not exist, skipping.")
            continue
        embeddings = []
        for img_name in os.listdir(instance_path):
            img_path = os.path.join(instance_path, img_name)
            emb = extract_embeddings(img_path, app_instances[0])
            if emb is not None:
                embeddings.append(emb)
        avg_emb = compute_average_embedding(embeddings)
        if avg_emb is not None:
            gt_embeddings[instance_id] = avg_emb
        else:
            print(f"No valid embeddings for Ground Truth instance: {instance_id}, skipping.")
    print(f"Loaded Ground Truth embeddings for {len(gt_embeddings)} instances.")
    return gt_embeddings

def process_instance(instance_folder, gt_embeddings, gpu_count):
    """
    处理每个生成图片的实例文件夹：
    - 提取instance_id。
    - 检查生成图片中是否有100张图片。
    - 对每个prompt（5张图片），提取检测到的人脸嵌入并计算平均嵌入向量。
    - 计算与Ground Truth的余弦相似度。
    - 仅当至少一个prompt有有效相似度时，返回结果。
    """
    folder_name = os.path.basename(instance_folder)
    if '_reg' in folder_name:
        instance_id = folder_name.split('_reg')[0]
    else:
        print(f"Folder name '{folder_name}' does not contain '_reg', skipping.")
        return None

    gt_emb = gt_embeddings.get(instance_id)
    if gt_emb is None:
        print(f"No Ground Truth embedding for instance: {instance_id}, skipping.")
        return None

    samples_folder = os.path.join(instance_folder, 'samples')
    if not os.path.isdir(samples_folder):
        print(f"No samples folder in {instance_folder}, skipping.")
        return None

    image_files = [f"{i}.jpg" for i in range(100)]
    existing_image_files = [img for img in image_files if os.path.exists(os.path.join(samples_folder, img))]
    if len(existing_image_files) < 100:
        print(f"Not all 100 images exist in {samples_folder}, skipping.")
        return None

    gpu_id = hash(instance_id) % gpu_count
    app = app_instances[gpu_id]

    prompt_similarities = []
    valid_prompt_count = 0

    for i in range(0, 100, 5):
        prompt_images = [f"{i+j}.jpg" for j in range(5)]
        prompt_embeddings = []
        for img_name in prompt_images:
            img_path = os.path.join(samples_folder, img_name)
            emb = extract_embeddings(img_path, app)
            if emb is not None:
                prompt_embeddings.append(emb)
        if prompt_embeddings:
            prompt_avg_emb = compute_average_embedding(prompt_embeddings)
            if prompt_avg_emb is not None:
                similarity = np.dot(prompt_avg_emb, gt_emb)
                prompt_similarities.append(similarity)
                valid_prompt_count += 1

    if valid_prompt_count == 0:
        print(f"No valid prompts with detected faces for instance: {instance_id}, skipping.")
        return None

    overall_similarity = np.mean(prompt_similarities)

    return {
        'instance_id': instance_id,
        'prompt_similarities': prompt_similarities,
        'overall_similarity': overall_similarity
    }

def main(generated_root, gt_root, output_csv, num_workers):
    gpu_count = torch.cuda.device_count()
    num_workers = min(num_workers, gpu_count)
    print(f"Initializing {gpu_count} FaceAnalysis instances for GPU processing...")
    initialize_app_instances(gpu_count)

    print("Extracting generated instances IDs...")
    generated_instance_folders = [os.path.join(generated_root, d) for d in os.listdir(generated_root) 
                                  if os.path.isdir(os.path.join(generated_root, d))]
    generated_ids = []
    for folder in generated_instance_folders:
        folder_name = os.path.basename(folder)
        if '_reg' in folder_name:
            instance_id = folder_name.split('_reg')[0]
            generated_ids.append(instance_id)
        else:
            print(f"Folder name '{folder_name}' does not contain '_reg', skipping.")

    gt_embeddings = load_ground_truth_embeddings(gt_root, generated_ids)

    valid_generated_folders = [folder for folder in generated_instance_folders 
                               if os.path.basename(folder).split('_reg')[0] in gt_embeddings]

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for instance_folder in valid_generated_folders:
            futures.append(executor.submit(process_instance, instance_folder, gt_embeddings, gpu_count))
        
        for future in tqdm(futures, desc="Processing instances"):
            result = future.result()
            if result is not None:
                results.append(result)

    if not results:
        print("No valid results to write.")
        return

    max_prompts = max(len(res['prompt_similarities']) for res in results)
    prompt_columns = [f"prompt_{i+1}" for i in range(max_prompts)]
    fieldnames = ['instance_id'] + prompt_columns + ['overall_similarity']

    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            row = {'instance_id': res['instance_id']}
            for i, sim in enumerate(res['prompt_similarities']):
                row[f"prompt_{i+1}"] = sim
            row['overall_similarity'] = res['overall_similarity']
            writer.writerow(row)

    avg_similarities = {}
    overall_similarities = []
    for res in results:
        for i, sim in enumerate(res['prompt_similarities']):
            prompt_key = f"prompt_{i+1}"
            if prompt_key not in avg_similarities:
                avg_similarities[prompt_key] = []
            avg_similarities[prompt_key].append(sim)
        overall_similarities.append(res['overall_similarity'])
    
    overall_avg = {}
    for prompt_key, sims in avg_similarities.items():
        overall_avg[prompt_key] = np.mean(sims) if sims else 0.0
    overall_avg['overall_similarity'] = np.mean(overall_similarities) if overall_similarities else 0.0

    with open(output_csv, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        avg_row = {'instance_id': 'average'}
        for key, value in overall_avg.items():
            avg_row[key] = value
        writer.writerow(avg_row)

    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated face images against Ground Truth.")
    parser.add_argument("--generated_root", type=str, required=True, help="Path to the generated images root folder")
    parser.add_argument("--gt_root", type=str, required=True, help="Path to the Ground Truth dataset folder")
    parser.add_argument("--output_csv", type=str, default="evaluation_results.csv", help="Path to the output CSV file")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker threads")
    
    args = parser.parse_args()
    main(args.generated_root, args.gt_root, args.output_csv, args.num_workers)
