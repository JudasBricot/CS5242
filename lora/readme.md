# LoRA Project

## Overview
This project is built upon Stable Diffusion and leverages LoRA technique for fine-tuning the model with custom concepts.

## Environment Setup
To get started with the project, follow these steps to configure the environment:

1. **Stable Diffusion Setup**: This project is based on Stable Diffusion version 1.5. You need to set up a conda environment for the Latent Diffusion Model (LDM) using the requirements provided for SD v1.5.
   ```bash
   # creating conda environment
   git clone https://github.com/CompVis/stable-diffusion.git
   cd stable-diffusion
   conda env create -f environment.yaml
   ```

2. **Install Diffusers**: The Diffusers library must be installed from the source. Clone the Diffusers repository and install it manually:
   ```bash
   # conda activate ldm
   git clone https://github.com/huggingface/diffusers.git
   cd diffusers
   pip install -e .
   pip install -r examples/textual_inversion/requirements.txt
   ```

3. **Download Weights**: You will need the pre-trained weights for Stable Diffusion v1.5 from huggingface. The default weights should be saved at `runwayml/stable-diffusion-v1-5`.

## Dataset Preparation
The project uses the **custom-101** dataset. The preprocessing is done by the `create_formated_dataset.py`script. This step is included in the `train_sample.sh` script.

## Training Script
The training is conducted using the script `train_text_to_image_lora.py`. To simplify running the script, use the provided shell script `train_sample.sh`. Before starting training, you need to make a few modifications:

- Update the dataset path to point to the location of your **custom-101** dataset concepts.
- Optionally, adjust the path for saving the trained model files.

## Image Generation
To generate images with the trained model, is done in the script `generate_image_example.py`, but we provide the `generate_sample.sh` file to ease the generation. Make the following changes to the script before running:

- Modify the `prompts` in the script to match the evaluation requirements for testing the trained model.
- Optionnaly, update the name of the directory of output.
- Optionnaly, update the `lora_weights_dir` path to use the checkpoint generated from the training process.

## Running the Scripts
- **Training**: Execute `train_sample.sh` after making the necessary adjustments.
- **Generation**: Execute `generate_sample.sh` after making the necessary adjustments.

Feel free to adjust any other parameters to suit your specific requirements.