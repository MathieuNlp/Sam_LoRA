
from src.segment_anything.utils.transforms import ResizeLongestSide
import numpy as np
import torch
from typing import Optional, Tuple
import einops
from einops import rearrange

class Samprocessor:
    def __init__(self, sam_model):
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

    def __call__(self, image, original_size, prompt):
        # Processing of the image
        image_torch = self.process_image(image, original_size)
        # embeddings dans self.features
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
        # Transform input prompts
        box_torch = self.process_prompt(prompt, original_size)
        inputs = {"image": image_torch, 
                  "original_size": original_size,
                 "boxes": box_torch}
        
        return inputs


    def process_image(self, image, original_size):
        nd_image = np.array(image)
        input_image = self.transform.apply_image(nd_image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        processed_image = self.model.preprocess(input_image_torch) 
        self.is_image_set = True

        return processed_image

    def process_prompt(self, box, original_size):
        # We only use boxes
        box_torch = None
        nd_box = np.array(box)
        box = self.transform.apply_boxes(nd_box, original_size)
        box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
        box_torch = box_torch[None, :]

        return box_torch


    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None