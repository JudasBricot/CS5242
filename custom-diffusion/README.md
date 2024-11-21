# Subject Centric Image Generation

Project goal: to generate subject in images based on a few photos of them. Subjects in this case refer to people.

The codebase is a merge of [Multi-Concept Customization of Text-to-Image Diffusion (Custom Diffusion)](https://github.com/adobe-research/custom-diffusion) and [Cross Initialisation for Face Personalisation of Text-to-Image Models (Cross initialisation)](https://github.com/lyuPang/CrossInitialization). Their original READMEs are shown below.

## Experiment

For all setups, the base model is stable-diffusion v1.5

We run the code with different setups:

- **Custom diffusion (CD) with no regularisation data**: trained with no regularisation data. Run it with `pipeline_no_reg.sh`
- **Custom diffusion (CD) with generated regularisation data**: trained with generated data as regularisation. The data is generated using the base model. Run it with `pipeline_reg.sh`. Will run longer because data need to be generated first.
- **Custom diffusion (CD) with alternative generated regularisation data**: trained with another set of generated data as regularisation. The model is generated using a 'higher fidelity' model: juggernautXI (a finetuned SDXL model) with bad_quality lora. Run it with `pipeline_alt_reg.sh`
- **Cross Intialization (CR) with textual embedding**: trained with Cross initialisation method -  a form of textual inversion method except the modifier tokens are trained with the average embeddings of celebrities' faces. Run it with `pipeline_cr.sh`
- **CD + CR with generated regularisation data**: combination of Custom Diffusion and Cross initialisation. CD finetune the cross attention and CR finetune the text embeddings. Run it with `pipeline_reg_cr.sh`. Will run longer because data need to be generated first.
- You will need to update the corresponding directory and `batch_eval_output_*.json`

Common parameters:

- CR and CD parameters are trained as recommended in their git README.
  - CD is trained with the recommended setting of 2 GPU. One can run with 1 GPU but the learning rate needs to be tuned. 
- For faces, CD's authors suggest the following parameters;
  - max_train_steps: 1500
  - learning_rate: 5e-6
  - freeze_model: crossattn

## Installation

The experiment runs on Python 3.10. Packages are stated in `requirements.txt`. diffusers requirement are quite fixed, upgrading it will be a problem. Note, while deepspeed is installed, it does not work on this. Accelerate config is run using bf16, no deepspeed.

# Original repositories

[Custom Diffusion](https://github.com/adobe-research/custom-diffusion)

[Cross Initialization](https://github.com/lyuPang/CrossInitialization)