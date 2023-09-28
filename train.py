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

# Load the config file
with open("./config.yaml", "r") as ymlfile:
   config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Take dataset path
train_dataset_path = config_file["DATASET"]["TRAIN_PATH"]

# Load SAM model
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
#Create SAM LoRA
sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])  
model = sam_lora.sam
# model = sam
# for name, param in model.named_parameters():
#   if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
#     param.requires_grad_(False)

# Process the dataset
processor = Samprocessor(model)
dataset = DatasetSegmentation(config_file, processor)

# Create a dataloader
train_dataloader = DataLoader(dataset, batch_size=config_file["TRAIN"]["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn)

# Initialize optimize and Loss
optimizer = Adam(model.image_encoder.parameters(), lr=1e-4, weight_decay=0)

seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set model to train and into the device
model.train()
model.to(device)


for epoch in range(num_epochs):
    epoch_losses = []
    for i, batch in enumerate(tqdm(train_dataloader)):
      
      outputs = model(batched_input=batch,
            multimask_output=False)
      
      
      list_gt_msk, list_pred_msk, list_bbox = utils.get_list_masks(batch, outputs)
      if epoch % 10 == 0:
        utils.tensor_to_image(list_gt_msk, list_pred_msk, list_bbox, i, config_file["TRAIN"]["BATCH_SIZE"])

      gt_mask_tensor = batch[0]["ground_truth_mask"].unsqueeze(0).unsqueeze(0) # We need to get the [B, C, H, W] starting from [H, W]
      loss = seg_loss(outputs[0]["low_res_logits"], gt_mask_tensor.float().to(device))
      
      
      optimizer.zero_grad()
      loss.backward()

      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())
         

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')

# Save the parameters of the model in safetensors format
sam_lora.save_lora_parameters("lora.safetensors")

