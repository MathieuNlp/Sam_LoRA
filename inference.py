import torch
from src.segment_anything import build_sam_vit_b
from src.processor import Samprocessor
from src.lora import LoRA_sam
from PIL import Image
import matplotlib.pyplot as plt
import src.utils as utils

dataset_path = "./bottle_glass_dataset"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Path of the image we test
image_path = "./bottle_glass_dataset/test/perfume2.jpg"
bbox = [307, 107, 646, 925]

# Load LoRA SAM model
sam = build_sam_vit_b(checkpoint="sam_vit_b_01ec64.pth")
sam_lora = LoRA_sam(sam,4)
sam_lora.load_lora_parameters("lora.safetensors")
with torch.no_grad():
    sam_lora.sam.eval()

    # Process the image
    processor = Samprocessor(sam_lora.sam)
    image =  Image.open(image_path)
    original_size = tuple(image.size)[::-1]

    # Predict the mask
    inputs = [processor(image, original_size, bbox)]
    outputs = sam_lora.sam(inputs, multimask_output=False)

    # Plot the mask
    pred_mask = outputs[0]["masks"].squeeze(0).squeeze(0).numpy()
    pred_mask_pil = Image.fromarray(pred_mask)
    utils.plot_image_mask(image, pred_mask_pil, "perfume2")
