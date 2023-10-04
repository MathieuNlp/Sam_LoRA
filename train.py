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
from src.model_checkpoint import ModelCheckpoint

# Load the config file
with open("./config.yaml", "r") as ymlfile:
   config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Take dataset path
train_dataset_path = config_file["DATASET"]["TRAIN_PATH"]
train_dataset_path = config_file["DATASET"]["VALID_PATH"]
# Load SAM model
sam = build_sam_vit_b(checkpoint=config_file["SAM"]["CHECKPOINT"])
#Create SAM LoRA
sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])  
model = sam_lora.sam
# Process the dataset
processor = Samprocessor(model)
train_ds = DatasetSegmentation(config_file, processor, mode="train")
valid_ds = DatasetSegmentation(config_file, processor, mode="valid")
# Create a dataloader
train_dataloader = DataLoader(train_ds, batch_size=config_file["TRAIN"]["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
# Initialize optimize and Loss
optimizer = Adam(model.image_encoder.parameters(), lr=1e-4, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]

model_checkp = ModelCheckpoint(sam_lora)

device = "cuda" if torch.cuda.is_available() else "cpu"
# Set model to train and into the device
model.to(device)

total_loss = []

for epoch in range(num_epochs):
    epoch_losses = []
    model.train()

    for i, batch in enumerate(tqdm(train_dataloader)):
      
      outputs = model(batched_input=batch,
                      multimask_output=False)
      
      gt_mask_tensor = batch[0]["ground_truth_mask"].unsqueeze(0).unsqueeze(0) # We need to get the [B, C, H, W] starting from [H, W]
      loss = seg_loss(outputs[0]["low_res_logits"], gt_mask_tensor.float().to(device))
      
      optimizer.zero_grad()
      loss.backward()
      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())
         
    model.eval()
    valid_loss = 0
    for i, valid_batch in enumerate(tqdm(valid_dataloader)):
       
      valid_outputs = model(batched_input=valid_batch,
                      multimask_output=False)
      gt_mask_tensor = valid_batch[0]["ground_truth_mask"].unsqueeze(0).unsqueeze(0) # We need to get the [B, C, H, W] starting from [H, W]
      valid_loss += seg_loss(valid_outputs[0]["low_res_logits"], gt_mask_tensor.float().to(device))
    model_checkp.update(valid_loss, epoch)
    print(valid_loss)

    print(f'EPOCH: {epoch}')
    print(f'Mean loss training: {mean(epoch_losses)}')
    print(f'Mean loss validation : {valid_loss/2}')

# Save the parameters of the model in safetensors format
rank = config_file["SAM"]["RANK"]
sam_lora.save_lora_parameters(f"lora_rank{rank}.safetensors")
