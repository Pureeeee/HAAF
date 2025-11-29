from itertools import repeat
import collections.abc

from torch import nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d
import torch
import torch.nn.functional as F
import math

def resize_pos_embed(pos_embed, x):

    n_old = pos_embed.shape[1] - 1
    n_new = x.shape[1] - 1
    

    if n_new == n_old:
        return pos_embed


    cls_pos_embed = pos_embed[:, :1]  
    

    patch_pos_embed = pos_embed[:, 1:]  
    

    h_old = w_old = int(math.sqrt(n_old))
    h_new = w_new = int(math.sqrt(n_new))

    patch_pos_embed = patch_pos_embed.reshape(1, h_old, w_old, -1).permute(0, 3, 1, 2)
    

    patch_pos_embed = F.interpolate(
        patch_pos_embed,
        size=(h_new, w_new),
        mode='bicubic',
        align_corners=False,
    )
    

    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
    

    resized_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)
    
    return resized_pos_embed

def freeze_batch_norm_2d(module, module_match={}, name=''):

    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res



def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)
