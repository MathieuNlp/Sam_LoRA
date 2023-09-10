import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize

import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
from einops import rearrange

dataset_path = "./bottle_glass_dataset"

sam = build_sam_vit_b(checkpoint="sam_vit_b_01ec64.pth")
sam_lora = LoRA_sam(sam,4)  
model = sam_lora.sam
processor = Samprocessor(model)
dataset = DatasetSegmentation(dataset_path, processor)
#utils.plot_image_mask_dataset(dataset, 3)

train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
optimizer = Adam(sam_lora.lora_vit.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

num_epochs = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
model.train()


for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
      batch_inputs = [batch[0][0]]
      outputs = model(batched_input=batch_inputs,
            multimask_output=False)

      # compute loss
      gt_outputs_mask = outputs[0]["masks"].to(device)

      #predicted_masks = outputs.masks.squeeze(1)
      ground_truth_masks = batch_inputs[0]["ground_truth_mask"].float().to(device)
      ground_truth_masks = ground_truth_masks.contiguous()[None, None, :, :]
      print(ground_truth_masks.shape)
      ground_truth_masks = rearrange(ground_truth_masks, "b x w h -> b x h w")
      print(gt_outputs_mask.shape, ground_truth_masks.shape, ground_truth_masks[0].shape)
      loss = seg_loss(gt_outputs_mask, ground_truth_masks.unsqueeze(1)[0])

      # backward pass (compute gradients of parameters w.r.t. loss)
      optimizer.zero_grad()
      loss.backward()

      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())


    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')