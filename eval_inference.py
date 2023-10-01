import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize
from torchvision.utils import save_image
import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F

def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the config file
with open("./config.yaml", "r") as ymlfile:
   config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Load SAM model
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
#Create SAM LoRA
sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])
sam_lora.load_lora_parameters(f"./lora_weights/lora_rank{sam_lora.rank}.safetensors")  
model = sam_lora.sam

# Process the dataset
processor = Samprocessor(model)
dataset = DatasetSegmentation(config_file, processor, is_test=True)

# Create a dataloader
test_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)


# Set model to train and into the device
model.eval()
model.to(device)

with torch.no_grad():
    total_score = []
    for i, batch in enumerate(tqdm(test_dataloader)):
        
        outputs = model(batched_input=batch,
            multimask_output=False)

        gt_mask_tensor = batch[0]["ground_truth_mask"].unsqueeze(0).unsqueeze(0) # We need to get the [B, C, H, W] starting from [H, W]
        dice_score = dice_loss(outputs[0]["low_res_logits"], gt_mask_tensor.float().to(device))

        total_score.append(dice_score.item())
        print("Dice score: ", total_score[-1])
        

    print(f'Mean loss: {mean(total_score)}')
