import numpy as np
import torch
import lora 

class ModelCheckpoint:

    def __init__(self, model: lora.LoRA_sam):
        self.min_loss = None
        self.model = model
        self.saved_A_weights = None
        self.saved_B_weights = None

    def update(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            self.min_loss = loss
            self.saved_A_weights = self.model.A_weights
            self.saved_B_weights = self.model.B_weights

            