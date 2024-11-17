# Evaluation Methods for Image-Text and Face Alignment

This repository provides two evaluation methods for assessing image-text alignment and face alignment in generated datasets:

1. **`evaluate_clip`**: Uses CLIP to evaluate text-to-image alignment.
2. **`evaluate_face`**: Uses InsightFace's SCRFD to assess face alignment.

## Installation Requirements

To use the provided evaluation methods, please ensure you have the following libraries installed:

- [InsightFace](https://github.com/deepinsight/insightface/tree/master/python-package)
- `onnxruntime-gpu`

If you encounter issues running the face alignment evaluation on GPU, you can use the provided `environment.yml` file to set up the necessary conda environment.

## Usage Instructions

### 1. Image-Text Alignment: `evaluate_clip`

To evaluate image-text alignment using CLIP, run the following command:

```bash
python evaluate_clip.py --root_dir <path_to_root_dir> --prompt_file <path_to_prompt_file>
```

- **`--root_dir`**: The root directory should follow the structure `id/samples`, where `id` is a unique identifier for each face instance, and `samples` contains 100 images, with each 5 images corresponding to a specific prompt.
- **`--prompt_file`**: Path to the prompt file, which contains the prompts corresponding to each set of 5 images.
- **`--append_prefix`** and **`--prefix`** (optional): These parameters can be used to customize the text prefix. By default, the prefix is set to "A photo depicts".

### 2. Face Alignment Evaluation: `evaluate_face`

To evaluate face alignment using InsightFace's SCRFD method, run the following command:

```bash
python evaluate_face.py --generated_root <path_to_generated_root> --gt_root <path_to_ground_truth_root> --output_csv <output_csv_path>
```

- **`--generated_root`**: The root directory containing generated images, following the same format as `root_dir` in `evaluate_clip`, with `id/samples`.
- **`--gt_root`**: The ground truth directory containing face images. Each subdirectory should correspond to an `id` and contain several photos of the same face.
- **`--output_csv`**: Specifies the path to save the output CSV file containing the face alignment evaluation results.

This method requires the use of InsightFace. Refer to the [InsightFace GitHub repository](https://github.com/deepinsight/insightface/tree/master/python-package) for more details on setting up the necessary dependencies.

