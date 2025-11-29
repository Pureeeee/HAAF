import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from conch.open_clip_custom import resize_pos_embed
class ConchAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(ConchAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)

        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )
    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y

class CONCH_Inplanted_15(nn.Module):
    def __init__(self, conch_v15_model, features):
        super().__init__()

        self.image_encoder = conch_v15_model
        self.features = features
        embed_dim = self.image_encoder.embed_dim
        self.det_adapters = nn.ModuleList( [ConchAdapter(c_in = embed_dim, bottleneck=768) for i in range(len(features))] )

    def forward(self, x):

        x = self.image_encoder.patch_embed.proj(x)
  
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)  
        
        x =  torch.cat(
            [self.image_encoder.cls_token.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)
        

        pos_embed = resize_pos_embed(self.image_encoder.pos_embed, x)

        x = x + pos_embed.to(x.dtype)
        x = self.image_encoder.pos_drop(x)

        
        det_patch_tokens = []



        
        for i, blk in enumerate(self.image_encoder.blocks):      


            x = blk(x)
            if (i + 1) in self.features:
                idx = self.features.index(i + 1)
                    

                det_adapt_med, det_adapt_out = self.det_adapters[idx](x)
                    

                x = 0.8 * x + 0.2 * det_adapt_out   


                det_patch_tokens.append(det_adapt_med)
            

        return det_patch_tokens