import torch
import monai
import numpy as np
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize

import src.utils as utils
from src.dataloader import TestDatasetSegmentation

from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor, sam_model_registry
from src.lora import LoRA_sam
import einops
from einops import rearrange

from PIL import Image

dataset_path = "./bottle_glass_dataset"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "vit_b"
sam_checkpoint = "sam_vit_b_01ec64.pth"

image_path = "./bottle_glass_dataset/images/perfume3.jpg"
mask_path = "./bottle_glass_dataset/masks/perfume3.tiff"

def inference_standard_sam(image_path, mask_path):

    image = Image.open(image_path)
    mask = Image.open(mask_path)
    mask = mask.convert('1')
    ground_truth_mask =  np.array(mask)
    box = utils.get_bounding_box(ground_truth_mask)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(np.array(image))
    input_box = utils.get_bounding_box(np.array(mask))
    masks, _, _ = predictor.predict(
        box=input_box,
        multimask_output=False,
    )
    print(mask)


inference_standard_sam(image_path, mask_path)

    

