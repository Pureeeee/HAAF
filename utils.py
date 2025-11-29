import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

import kornia.geometry.transform as KGeo  
import kornia.augmentation as KAug        

from PIL import Image

def cos_sim(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

def get_translation_mat(a, b):
    return torch.tensor([[1, 0, a],
                         [0, 1, b]])

def rot_img(x, theta):
    dtype = torch.FloatTensor
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    return x

def translation_img(x, a, b):
    dtype = torch.FloatTensor
    rot_mat = get_translation_mat(a, b)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    return x

def hflip_img(x):

    x = KGeo.hflip(x) 
    return x

def vflip_img(x):

    x = KGeo.vflip(x)
    return x

def rot90_img(x,k):
    degreesarr = [0., 90., 180., 270., 360]
    degrees = torch.tensor(degreesarr[k])

    x = KGeo.rotate(x, angle = degrees, padding_mode='reflection')
    return x


def augment(fewshot_img, fewshot_mask=None):
    augment_fewshot_img = fewshot_img

    if fewshot_mask is not None:
        augment_fewshot_mask = fewshot_mask

 
        for angle in [-np.pi/8, -np.pi/16, np.pi/16, np.pi/8]:
            rotate_img = rot_img(fewshot_img, angle)
            augment_fewshot_img = torch.cat([augment_fewshot_img, rotate_img], dim=0)

            rotate_mask = rot_img(fewshot_mask, angle)
            augment_fewshot_mask = torch.cat([augment_fewshot_mask, rotate_mask], dim=0)

        for a,b in [(0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
            trans_img = translation_img(fewshot_img, a, b)
            augment_fewshot_img = torch.cat([augment_fewshot_img, trans_img], dim=0)

            trans_mask = translation_img(fewshot_mask, a, b)
            augment_fewshot_mask = torch.cat([augment_fewshot_mask, trans_mask], dim=0)


        flipped_img = hflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)
        flipped_mask = hflip_img(fewshot_mask)
        augment_fewshot_mask = torch.cat([augment_fewshot_mask, flipped_mask], dim=0)


        flipped_img = vflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)

        flipped_mask = vflip_img(fewshot_mask)
        augment_fewshot_mask = torch.cat([augment_fewshot_mask, flipped_mask], dim=0)

    else:

        for angle in [-np.pi/8, -np.pi/16, np.pi/16, np.pi/8]:
            rotate_img = rot_img(fewshot_img, angle)
            augment_fewshot_img = torch.cat([augment_fewshot_img, rotate_img], dim=0)


        for a,b in [(0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
            trans_img = translation_img(fewshot_img, a, b)
            augment_fewshot_img = torch.cat([augment_fewshot_img, trans_img], dim=0)


        flipped_img = hflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)

        flipped_img = vflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)

        B, _, H, W = augment_fewshot_img.shape
        augment_fewshot_mask = torch.zeros([B, 1, H, W])
    
    return augment_fewshot_img, augment_fewshot_mask

class DynamicAugment(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.aug = nn.Sequential(
            
            KAug.RandomHorizontalFlip(p=0.5),
            KAug.RandomVerticalFlip(p=0.5),
            KAug.RandomRotation(degrees=15.0, p=0.7),
            KAug.RandomAffine(degrees=0, translate=(0.1, 0.1), p=0.5),
            KAug.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            KAug.RandomSharpness(sharpness=0.5, p=0.3),
        )
        self.img_size = img_size

    @torch.no_grad()
    def forward(self, x):
        x_aug = self.aug(x)
        noise = torch.randn_like(x_aug) * 0.02 
        x_aug = x_aug + noise
        return torch.clamp(x_aug, 0, 1)