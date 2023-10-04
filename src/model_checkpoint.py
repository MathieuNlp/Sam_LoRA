import numpy as np
import torch


class ModelCheckpoint:

    def __init__(self, model):
        self.min_loss = None
        self.model = model
        self.epoch = None

    def update(self, loss, epoch):
        if (self.min_loss is None) or (loss < self.min_loss):
            self.min_loss = loss
            self.epoch = epoch

            