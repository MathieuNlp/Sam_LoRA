import torch
from src.segment_anything import build_sam_vit_b
from src.processor import Samprocessor
from src.lora import LoRA_sam
from PIL import Image
import matplotlib.pyplot as plt
import src.utils as utils
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("./config.yaml", "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Path of the image we test
image_path = config_file["TEST"]["IMAGE_PATH"]
bbox = config_file["TEST"]["BBOX"]

# Load LoRA SAM model
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])
sam_lora.load_lora_parameters("lora.safetensors")
model = sam_lora.sam
model.to(device)
with torch.no_grad():
    model.eval()

    # Process the image
    processor = Samprocessor(model)
    image =  Image.open(image_path)
    original_size = tuple(image.size)[::-1]

    # Predict the mask
    inputs = [processor(image, original_size, bbox)]
    outputs = model(inputs, multimask_output=False)

    # Plot the mask
    pred_mask = outputs[0]["masks"].squeeze(0).squeeze(0).cpu().numpy()
    pred_mask_pil = Image.fromarray(pred_mask)
    utils.plot_image_mask(image, pred_mask_pil, "ring4_test")
