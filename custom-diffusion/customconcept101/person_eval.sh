#!/bin/bash

sample_root=("/workspace/output/pet_cat1" \
            "/workspace/output/pet_cat2" \
            "/workspace/output/pet_cat4" \
            "/workspace/output/pet_cat5")

target_path=("/workspace/data/benchmark_dataset/pet_cat1" \
            "/workspace/data/benchmark_dataset/pet_cat2" \
            "/workspace/data/benchmark_dataset/pet_cat4" \
            "/workspace/data/benchmark_dataset/pet_cat5")

# python evaluate.py --sample_root /workspace/output/person_1/ --target_path /workspace/data/benchmark_dataset/person_1 --numgen 100
# python evaluate.py --sample_root /workspace/output/person_2/ --target_path /workspace/data/benchmark_dataset/person_2 --numgen 100
# python evaluate.py --sample_root /workspace/output/person_3/ --target_path /workspace/data/benchmark_dataset/person_3 --numgen 100

# s=("/workspace/data/benchmark_dataset/person_1" \
#             "/workspace/data/benchmark_dataset/person_2" \
#             "/workspace/data/benchmark_dataset/person_3")
# output_dirs=("/workspace/output/person_1" \
#             "/workspace/output/person_2" \
#             "/workspace/output/person_3")

for i in "${!sample_root[@]}"; do
    sample_dir="${sample_root[i]}"
    target_dir="${target_path[i]}"
    # echo ${sample_dir} ${target_dir} ${sample_dir}/evaluate.pkl
    python evaluate.py --sample_root ${sample_dir} --target_path ${target_dir} --numgen 100 --outpkl ${sample_dir}/evaluate.pkl
done