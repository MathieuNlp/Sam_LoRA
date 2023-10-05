# Segment Your Ring (SYR) - Segment Anything to segment rings (jewelery)

# Introduction

Segment anything is a foundational model released by Meta ([SAM](https://segment-anything.com/)). Pre-trained over 1 billion images, the model shows high performance zero-shot inference.

![Dog segmentation](./docs/images/dog_segmented.png)
*Segmentation of an image by SAM*

As good as they are, pre-trained models cannot answer to every segmentation tasks. The model has a large understanding of everything but situation-wise, it can be compromised. I would like to dig deeper in this problem by applying SAM to product packshots. 
Product packshots are mainly done with a unicolor background and the object upfront. Those images tends to have less noise and SAM should be performing really well in contrast of some city image with lots of information. However, we will that it's still a challenging problem.

# Problem
Can we segment jewelery rings used for product packshots ?

# Dataset
I built a dataset of 2 different rings with white background. The 2 types of rings are:
- Single ring
- Pair rings
The single rings have different views and a jewel in it. We can find silver and gold single rings.
The pair rings are 2 rings with one on top of the other. The outline of this can be challenging to segment.
In the training set, I tried to equally split both type of rings. The test set is constitued of 2 images, a single ring and a pair rings.
The dataset for train and test are in :
```
   ./dataset
```
![Test set](./docs/images/test_set.png)
*Test set*

# Baseline SAM
The SAM model has 3 blocks: The image encoder, prompt encoder and mask decoder. The mask decoder takes has input the encoded image and encoded prompt to return masks. 

![SAM Architecture Paper](./docs/images/sam_paper.png)
*SAM architecture from paper: https://arxiv.org/abs/2304.02643*

To get our baseline with the dataset, we will first see the capabilities of SAM with zero-shot inference. 

![Baseline preds test set](./docs/images/baseline_test_set_prediction.png)
*Baseline SAM predictions on test set*

![Baseline preds train set](./docs/images/baseline_train_set_prediction.png)
*Baseline SAM predictions on training set*

As we can see, SAM struggles to segment the ring. The model takes the inside of the ring has part of the object which is wrong. Now that we have assed the baseline model, how could we solve this issue ?

# Adapters



















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
If there is an error with Pytorch, Safetensors or CV2, do:
```sh
   poetry run pip install opencv-python safetensors torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```



## Get the SAM checkpoint (must be done inside "sam_lora_poetry" folder)
```sh
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

# Training
The training is not using the SamPredictor from Meta because I would like to be able to handle batches. So i created a processor.py file in /src that processes the images and prompts accordingly. The saved weights are in lora.safetensors.
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


# Acknowledgments
Thank you to:
- Niels Rogge - HuggingFace: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb
- JamesQFreeman: https://github.com/JamesQFreeman/Sam_LoRA
- Denis Brulé, Benjamin Trom - Finegrain AI

# Author
Mathieu Nalpon

