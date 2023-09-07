import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import json
import glob
import os 
from PIL import Image
import numpy as np

import src.utils as utils
from transformers import SamProcessor

dataset_path = "./bottle_glass_dataset"

class DatasetSegmentation(Dataset):
    def __init__(self, folder_path, processor, transform):
        super(DatasetSegmentation, self).__init__()
        # data processor of images bedore inputing into the SAM model
        self.processor = processor
        # get image and mask paths
        self.img_files = glob.glob(os.path.join(folder_path,'images','*.jpg'))

        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_path,'masks', os.path.basename(img_path)[:-4] + ".tiff")) 
        self.transform = transform

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            # get image and mask in PIL format
            image =  Image.open(img_path)
            mask = Image.open(mask_path)
            mask = mask.convert('1')
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)  
            ground_truth_mask =  np.array(mask)

            # get bounding box prompt
            prompt = utils.get_bounding_box(ground_truth_mask)
            # process image and masks with the processor
            inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

            # remove batch dimension which the processor adds by default
            inputs = {k:v.squeeze(0) for k,v in inputs.items()}

            # add ground truth segmentation
            inputs["ground_truth_mask"] = ground_truth_mask

            return inputs



if __name__ == "__main__":
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    dataset = DatasetSegmentation(dataset_path, processor)
    #dataloader = DataLoader(dataset, batch_size=2, shuffle=True)