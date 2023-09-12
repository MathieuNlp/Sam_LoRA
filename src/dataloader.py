import torch
import glob
import os 
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import src.utils as utils

dataset_path = "./bottle_glass_dataset"

class DatasetSegmentation(Dataset):

    def __init__(self, folder_path, processor):
        super().__init__()
        # data processor of images bedore inputing into the SAM model
        # get image and mask paths
        self.img_files = glob.glob(os.path.join(folder_path,'images','*.jpg'))

        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_path,'masks', os.path.basename(img_path)[:-4] + ".tiff")) 

        self.processor = processor

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

            original_size = tuple(image.size)
    
            # get bounding box prompt
            box = utils.get_bounding_box(ground_truth_mask)
            inputs = self.processor(image, original_size, box)
            inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask)

            return inputs
    
def collate_fn(batch):
    return list(batch)