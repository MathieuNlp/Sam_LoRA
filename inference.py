import torch
import numpy as np 
from src.segment_anything import build_sam_vit_b, SamPredictor, sam_model_registry
from src.processor import Samprocessor
from src.lora import LoRA_sam
from PIL import Image
import matplotlib.pyplot as plt
import src.utils as utils
from PIL import Image, ImageDraw
import yaml
import json
from torchvision.transforms import ToTensor

sam_checkpoint = "sam_vit_b_01ec64.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_sam_baseline(checkpoint):
    sam = build_sam_vit_b(checkpoint=checkpoint)

    return sam

def load_sam_lora(sam_model, rank):
    sam_lora = LoRA_sam(sam_model, rank)
    sam_lora.load_lora_parameters(f"./lora_weights/lora_rank{rank}.safetensors")

    return sam_lora


def inference_model(sam_model, image_path, filename, mask_path=None, bbox=None, is_baseline=False):
    if is_baseline == False:
        model = sam_model.sam
        rank = sam_model.rank
    else:
        model = sam_model

    model.eval()
    model.to(device)
    image = Image.open(image_path)
    if mask_path != None:
        mask = Image.open(mask_path)
        mask = mask.convert('1')
        ground_truth_mask =  np.array(mask)
        box = utils.get_bounding_box(ground_truth_mask)
    else:
        box = bbox

    predictor = SamPredictor(model)
    predictor.set_image(np.array(image))
    masks, iou_pred, low_res_iou = predictor.predict(
        box=np.array(box),
        multimask_output=False,
    )

    if mask_path == None:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 15))
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline ="red")
        ax1.imshow(image)
        ax1.set_title(f"Original image + Bounding box: {filename}")

        ax2.imshow(masks[0])
        if is_baseline:
            ax2.set_title(f"Baseline SAM prediction: {filename}")
        else:
            ax2.set_title(f"SAM LoRA rank {rank} prediction: {filename}")
        plt.savefig("./plots/" + filename)

    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 15))
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline ="red")
        ax1.imshow(image)
        ax1.set_title(f"Original image + Bounding box: {filename}")

        ax2.imshow(ground_truth_mask)
        ax2.set_title(f"Ground truth mask: {filename}")

        ax3.imshow(masks[0])
        if is_baseline:
            ax3.set_title(f"Baseline SAM prediction: {filename}")
        else:
            ax3.set_title(f"SAM LoRA rank {rank} prediction: {filename}")
        plt.savefig("./plots/" + filename)


# Open configuration file
with open("./config.yaml", "r") as ymlfile:
   config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Open annotation file
f = open('annotations.json')
annotations = json.load(f)

sam_baseline_model = load_sam_baseline(sam_checkpoint)
sam_lora_model = load_sam_lora(sam_baseline_model, rank=4)

train_set = annotations["train"]
test_set = annotations["test"]
inference_train = False

if inference_train:
    total_loss = []
    for image_name, dict_annot in train_set.items():
        image_path = f"./dataset/train/images/{image_name}"
        inference_model(sam_lora_model, image_path, filename=image_name, mask_path=dict_annot["mask_path"], bbox=dict_annot["bbox"], is_baseline=False)


else:
    total_loss = []
    for image_name, dict_annot in test_set.items():
        image_path = f"./dataset/test/images/{image_name}"
        inference_model(sam_lora_model, image_path, filename=image_name, bbox=dict_annot["bbox"], is_baseline=False)
        
        
