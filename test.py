import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize

import src.utils as utils
from src.dataloader import TestDatasetSegmentation

from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import einops
from einops import rearrange
dataset_path = "./bottle_glass_dataset"

ds = TestDatasetSegmentation(dataset_path)
train_dataloader = DataLoader(ds, batch_size=2)

sam = build_sam_vit_b(checkpoint="sam_vit_b_01ec64.pth")
sam_lora = LoRA_sam(sam,4)  
model = sam_lora.sam

optimizer = Adam(sam_lora.lora_vit.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

num_epochs = 20

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
model.train()


for epoch in range(num_epochs):
    epoch_losses = []
    for image, mask, box in tqdm(train_dataloader):
      # inv_batch = utils.dict_list_inversion(batch)
      # outputs = model(batched_input=inv_batch,
      #       multimask_output=False)
      sampred = SamPredictor(model)
      image = rearrange(image, "x h w c -> h w (c x)")
      image = image.numpy()
      box = box.numpy()

               
      sampred.set_image(image)
      outputs = sampred.predict(box=box,
                            multimask_output=False)
      # compute loss
      predicted_masks = outputs[0]

      ground_truth_masks = mask.float().to(device)

      loss = seg_loss(torch.from_numpy(predicted_masks).float().to(device), ground_truth_masks)
      loss.requires_grad = True
      # backward pass (compute gradients of parameters w.r.t. loss)
      optimizer.zero_grad()
      loss.backward()

      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())


    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')