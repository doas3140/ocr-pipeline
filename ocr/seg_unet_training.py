# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/11_seg_unet_training.ipynb (unless otherwise specified).

__all__ = ['codes', 'train_transforms', 'valid_transforms', 'transforms', 'acc_camvid', 'name2id', 'void_code']

# Cell
from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
from .seg_dataset_isri_unlv import isri_unlv_config
from pathlib import PosixPath

# Cell
codes = list(isri_unlv_config.cat2id.keys())

# Cell
train_transforms = [
#     crop_pad(),
    rotate(degrees=(-10, 10), p=0.9),
    symmetric_warp(magnitude=(-0.1, 0.1), p=0.9),
#     dihedral_affine(p=1), # (flips image), will cause problems, because top left corner will be for example bottom right
#     rand_zoom(scale=(.5,1.), p=0.9),
    brightness(change=(0.4, 0.6), p=0.8),
    contrast(scale=(0.8,1.2), p=0.8),
]

valid_transforms = [
    rotate(degrees=(-1, 1), p=0.2)
]
transforms = (train_transforms, valid_transforms)
# transforms = get_transforms()

# Cell
name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Background']

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()