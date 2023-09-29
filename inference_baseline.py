import torch
import numpy as np
import src.utils as utils
from src.segment_anything import build_sam_vit_b, SamPredictor, sam_model_registry
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "vit_b"
sam_checkpoint = "sam_vit_b_01ec64.pth"

f = open('annotations.json')
annotations = json.load(f)

def inference_standard_sam(image_path, filename, mask_path=None, bbox=None):

    image = Image.open(image_path)
    if mask_path != None:
        mask = Image.open(mask_path)
        mask = mask.convert('1')
        ground_truth_mask =  np.array(mask)
        box = utils.get_bounding_box(ground_truth_mask)
        print(box)
        
    
    else:
        box = bbox
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(np.array(image))

    masks, _, _ = predictor.predict(
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
        ax2.set_title(f"Baseline SAM prediction: {filename}")
        plt.savefig("./plots/baseline/" + filename)

    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 15))
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline ="red")
        ax1.imshow(image)
        ax1.set_title(f"Original image + Bounding box: {filename}")

        ax2.imshow(ground_truth_mask)
        ax2.set_title(f"Ground truth mask: {filename}")

        ax3.imshow(masks[0])
        ax3.set_title(f"Baseline SAM prediction: {filename}")
        plt.savefig("./plots/baseline/" + filename)


train_set = annotations["train"]
test_set = annotations["test"]
inference_train = False

if inference_train:
    for image_name, dict_annot in train_set.items():
        image_path = f"./dataset/train/images/{image_name}"
        inference_standard_sam(image_path, filename=image_name, mask_path=dict_annot["mask_path"], bbox=dict_annot["bbox"])

else:
    for image_name, dict_annot in test_set.items():
        image_path = f"./dataset/test/{image_name}"
        inference_standard_sam(image_path, filename=image_name, bbox=dict_annot["bbox"])
