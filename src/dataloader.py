import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import json
import glob
import os 
from PIL import Image
import numpy as np

import utils
from transformers import SamProcessor

dataset_path = "./DIY_dataset"

class DatasetSegmentation(Dataset):
    def __init__(self, folder_path, processor):
        super(DatasetSegmentation, self).__init__()
        # data processor of images bedore inputing into the SAM model
        self.processor = processor
        # get image and mask paths
        self.img_files = glob.glob(os.path.join(folder_path,'images','*.jpg'))
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_path,'masks', os.path.basename(img_path)[:-4] + ".tiff")) 

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            # get image and mask in PIL format
            image =  Image.open(img_path)
            mask = Image.open(mask_path)
            mask = mask.convert('1')
            ground_truth_mask =  np.array(mask)

            # get bounding box prompt
            prompt = utils.get_bounding_box(ground_truth_mask)
            # process image and masks with the processor
            inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

            # remove batch dimension which the processor adds by default
            inputs = {k:v.squeeze(0) for k,v in inputs.items()}

            # add ground truth segmentation
            inputs["ground_truth_mask"] = ground_truth_mask
            inputs["pil_image"] = image
            inputs["pil_mask"] = mask

            return inputs


processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
dataset = DatasetSegmentation(dataset_path, processor)

# example = dataset[0]

# fig, axes = plt.subplots()
# print(example["ground_truth_mask"].shape)
# axes.imshow(np.array(example["pil_image"]))
# ground_truth_seg = np.array(example["pil_mask"])
# print(ground_truth_seg.shape)
# utils.show_mask(ground_truth_seg, axes)
# axes.title.set_text(f"Ground truth mask")
# axes.axis("off")