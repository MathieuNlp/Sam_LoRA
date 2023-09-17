from src.segment_anything import build_sam_vit_b, SamPredictor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from safetensors import safe_open
from safetensors.torch import save_file


class LoRA_qkv(nn.Module):

    def __init__(
            self,
            qkv,
            linear_a_q,
            linear_b_q,
            linear_a_v,
            linear_b_v,
    ):
        
        super(LoRA_qkv, self).__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.d_model = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)
        q_ba = self.linear_b_q(self.linear_a_q(x))
        v_ba = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, :self.d_model] += q_ba #q part
        qkv[:, :, :, -self.d_model:] += v_ba #v part

        return qkv


class LoRA_sam(nn.Module):

    def __init__(self, sam_model, rank, lora_layer=None):
        super(LoRA_sam, self).__init__()
        self.rank = rank
        assert rank > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            # In each block, you have an attention block => total blocks -> nb lora layers
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))
        
        self.A_weights = []
        self.B_weights = []

        # freeze parameters of the image encoder
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # if only lora on few layers
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            self.d_model = w_qkv_linear.in_features

            w_a_linear_q = nn.Linear(self.d_model, self.rank)
            w_b_linear_q = nn.Linear(self.rank, self.d_model)
            w_a_linear_v = nn.Linear(self.d_model, self.rank)
            w_b_linear_v = nn.Linear(self.rank, self.d_model)
            
            w_a_linear_q.requires_grad = True
            w_b_linear_q.requires_grad = True
            w_a_linear_v.requires_grad = True
            w_b_linear_v.requires_grad = True



            self.A_weights.append(w_a_linear_q)
            self.B_weights.append(w_b_linear_q)
            self.A_weights.append(w_a_linear_v)
            self.B_weights.append(w_b_linear_v)

            blk.attn.qkv = LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v
            )

        self.reset_parameters()
        self.sam = sam_model
        self.lora_vit = sam_model.image_encoder


    def reset_parameters(self):
        # Initalisation like in the paper
        for w_A in self.A_weights:
            nn.init.kaiming_uniform_(w_A.weight, a=np.sqrt(5))
        for w_B in self.B_weights:
            nn.init.zeros_(w_B.weight)


    def save_lora_parameters(self, filename):
        "save lora and fc parameters"
        num_layer = len(self.A_weights)
        # sufix 03:d -> allows to have a name 1 instead of 001
        a_tensors = {f"w_a_{i:03d}": self.A_weights[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.B_weights[i].weight for i in range(num_layer)}
        #dim_in = self.lora_vit.head.in_features
        #dim_out = self.lora_vit.head.out_features

        #fc_tensors = {f"fc_{dim_in}in_{dim_out}out": self.lora_vit.head.weight}

        #merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        merged_dict = {**a_tensors, **b_tensors}
        save_file(merged_dict, filename)


    def load_lora_parameters(self, filename):
        # load lora and fc parameters
        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.A_weights):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = nn.Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.B_weights):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = nn.Parameter(saved_tensor)

            #dim_in = self.lora_vit.head.in_features
            #dim_out = self.lora_vit.head.out_features
            #saved_key =  f"fc_{dim_in}in_{dim_out}out"

            #try:
            #    saved_tensors = f.get_tensor(saved_key)
            #    self.lora_vit.weight = nn.Parameter(saved_tensor)
            #except ValueError:
            #    print("This fc weight don't fit")


    def load_fc_parameters(self, filename):
        
        with safe_open(filename, framework="pt") as f:

            dim_in = self.lora_vit.head.in_features
            dim_out = self.lora_vit.head.out_features
            saved_key =  f"fc_{dim_in}in_{dim_out}out"

            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = nn.Parameter(saved_tensor)
            except ValueError:
                print("This fc don't fit")

            
if __name__ == "__main__":
    sam = build_sam_vit_b(checkpoint="sam_vit_b_01ec64.pth")
    sam_lora = LoRA_sam(sam,4)
    #sam_lora.sam.image_encoder(torch.rand(size=(1,3,1024,1024))
    print(sam_lora.parameters)