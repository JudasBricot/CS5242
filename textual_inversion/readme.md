# Textual Inversion Project

## Overview
This project is built upon Stable Diffusion and leverages textual inversion techniques for fine-tuning the model with custom embeddings.

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
The project uses the **custom-101** dataset. No preprocessing is required for this dataset at the moment. Simply make sure that the dataset is correctly structured and accessible.
Details can be found at https://github.com/adobe-research/custom-diffusion/tree/main/customconcept101

## Training Script
The training is conducted using the script `train_textual_inversion.py`. To simplify running the script, use the provided shell script `train_sample.sh`. Before starting training, you need to make a few modifications:

- Update the dataset path to point to the location of your **custom-101** dataset.
- Optionally, adjust the path for saving the checkpoint (ckpt) files during training.

## Image Generation
To generate images with the trained model, use the script `generate_image_example.py`. Make the following changes to the script before running:

- Update the `ckpt` path to use the checkpoint generated from the training process.
- Modify the `prompt` in the script to match the evaluation requirements for testing the trained model.

## Running the Scripts
- **Training**: Execute `train_sample.sh` after making the necessary adjustments.
- **Generation**: Use `generate_image_example.py` with updated checkpoint and prompt for generating evaluation images.

Feel free to adjust any other parameters to suit your specific requirements.

