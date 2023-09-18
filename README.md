# Segment Anything model fine tuned with LoRA for product packshots

In this project, we use the Segment Anything model released by Meta to capture masks of product packshots. This is challenging because some images can have shadows, reflections or even logos that needs to be taken into account.

## Segment Anything & LoRA

I chose the **vitb** image encoder. I applied LoRA to the attention modules inside the image encoder. I focused on queries and values as the LoRA (paper suggest that it is better).
I used bounding boxes for the input prompts.

# Setup
Get the repo with:
```sh
   git clone https://github.com/MathieuNlp/Sam_LoRA.git
```
# Demo
A gradio demo available. You can load your image and place 2 points to form a boudning box. After that run the generation of the mask.
```sh
   demo.ipynb
```
# Local Run

## Config file
There is a config file listing the hyperparameters to tune the model and some paths.
`
   config.yaml
`

## Poetry
All the dependecies are managed with poetry.
```sh
   cd sam_lora_poetry
   poetry install 
```

## Get the SAM checkpoint (must be done inside "sam_lora_poetry" folder)
```sh
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

# Training
The training is not using the SamPredictor from Meta because I would like to be alble the handle batches. So i created a processor.py file in /src that allows to process the image and prompt accordingly. The saved weights are in lora.safetensors.
```sh
   poetry run python train.py
```

# Inference
Run an inference with the saved weights from the training.
```sh
   poetry run python inference.py
```
# Plots
The plots folder regroup some comparaison and results to visualize the results.

- **comparaison.png**: Plot during the training the ground truth mask on top and predicted masks on the bottom.
- **gt_mask.jpg**: Ground truth mask example.
- **perfume2_notraining.jpg**: Perfume 2 mask predicted by the model with no training.
- **perfume2.jpg**: Perfume 2 mask predicted by the model trained with 10 epochs.
- **pred_perfume2_no_training.jpg**: Original image and predicted mask visualisation


# Sources
Thank you to:
- HuggingFace: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb
- JamesQFreeman: https://github.com/JamesQFreeman/Sam_LoRA

# Author
Mathieu Nalpon

