# Segment Your Ring (SYR) - Segment Anything to segment rings (jewelry)

# Introduction

Segment anything is a foundational model released by Meta ([SAM](https://segment-anything.com/)). Pre-trained over 1 billion images, the model shows high performance zero-shot inference.

![Dog segmentation](./docs/images/dog_segmented.png)
*Segmentation of an image by SAM*

As good as they are, pre-trained models cannot answer to every segmentation tasks. The model has a large understanding of everything but situation-wise, it can be compromised. I would like to dig deeper in this problem by applying SAM to product packshots. 
Product packshots are mainly done with a unicolor background and the object upfront. Those images tends to have less noise and SAM should be performing really well in contrast of some city image with lots of information. However, we will that it's still a challenging problem.

# Problem
Can we segment jewelry rings used for product packshots ?

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
![Baseline preds train set](./docs/images/baseline_train_prediction.png)
*Baseline SAM predictions on training set*


![Baseline preds test set](./docs/images/baseline_test_set_prediction.png)
*Baseline SAM predictions on test set*


As we can see, SAM struggles to segment the ring. The model takes the inside of the ring has part of the object which is wrong. In addition, it has trouble rightly segment the jewelry. To solve this problem, we can fine-tune the model. But there exists efficient ways to fine-tuned large pre-trained models.

# Adapters
The full fine-tuning process can be expensive, specially the bigger the model. An alternative for this is an adapter. Adapters plugs into chosen blocks of the frozen model and is trained. The training of adapters enable the adapted model to solve specific downstream task.This method can help solving our problem to a relatively low computing cost.

![SAM Architecture](./docs/images/sam_archi.png)
*SAM model architecture, source: Benjamin Trom - Finegrain AI*

For my model, I chose to use LoRA adapters.

## LoRA
LoRA is an adapter that is using 2 matrices B and A. The 2 matrices have specific dimensions (input_size, r) and (r, input_size) . By specefiying a rank r < input_size, we reduce the parameters size and try to capture the downstream task with a small enough rank. By doing the dot product B*A, we get a matrix of shape (input_size, input_size) so no information is lost. 

We only need to initialize the matrices, freeze SAM and tune the matrices components so that the frozen model + LoRA adapter model learns to segment rings.

# SAM LoRA
To apply LoRA to SAM, I had to choose a block to apply our adapter. I chose the image encoder because it could be interesting to tune the block that “understand”/encode the images. My LoRA implementation is adapting the **attention modules (queries and values)** of the ViT base by adding 2 nn.Linear in chain after computing queries and values (equivalent of B*A matrices product).

![SAM Architecture](./docs/images/sam_lora_archi.png)
*SAM + LoRA adaption*

Next I will explain the pipeline to train the model.

# Preprocessing
In the original model, a class “SamPredictor” is built to setup an image and predict. However, with this method, we can’t use a batching approach. Therefore, I created a class (Samprocessor) that preprocesses the datasets so that we can use batching for the training. 

The image go trough a longest stride resize and is normalized. Then the image is reshaped to 1024x1024 for the input encoder. Prompt needs to follow the resizing, therefore new coordinates are computed. The resized images and prompts are then passed to the mask decoder that outputs the mask with the highest IoU probability.

![Pre processing pipe](./docs/images/preprocessing_pipeline.png)
*Preprocessing pipeline*

Note: normalization of the image and reshape to 1024x1024 is done in:
```
   /src/segment_anything/modeling/sam
```
## Dataloader
```
   /src/dataloader.py
```
The Sam model requires as input a list(dict) object for batch training. To do this, I to created a dataloader that would generate this object. The dictionnary should contain 3 keys: 
- image: The processed image
- boxes: The processed prompt (here bounding box coordinates)
- original_size: The size of the image before transformation (used to transform the image back to the original size after predictions)

I added 2 keys:
- prompt: The bounding box coordinates before transformation
- ground_truth_mask: The ground truth mask

## Processor
```
   /src/processor.py
```
In the dataloader, the processor (Samprocessor class) tranforms the image and prompt so that both are prepared for the image encoder and prompt encoder. It will output the dictionnary containing the keys image, boxes and prompt.

![Processor inside](./docs/images/processor_inside.png)
*Processor transformation applied to image and prompt*


# Metrics
I used the Dice Loss to compute the results on the test set. By computing the dice loss, we have access to the Dice similarity coefficient (DSC) by doing: Dice coeff = 1 - Dice Loss.
The dice coefficient gauge the similarity of 2 samples. It is calculated from precision and call (similar to F1-score).
The loss is documented on this website: [Dice loss](https://docs.monai.io/en/stable/losses.html)

# Model selection

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

