# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00_core.ipynb (unless otherwise specified).

__all__ = ['test', 'save_inference', 'load_inference', 'plot', 'read_dict', 'save_dict']

# Cell
test = lambda: 'test'

# Cell
from fastai import *
from pathlib import PosixPath

def save_inference(learner, name, dir_path='../models'):
    temp_path = str(learner.path)
    learner.path = PosixPath(dir_path)
    learner.export(name)
    learner.path = PosixPath(temp_path)

# Cell
from fastai.vision import *

def load_inference(name, dir_path='../models/'):
    return load_learner(path=dir_path, file=name)

# example:
# save_learner(learner, 'unet')
# learner = load_learner_inference('unet')

# Cell
import matplotlib.pyplot as plt

def plot(im, figsize=None): # im - np.arr(h,w,3), figsize - tuple(2)
    ax = plt.figure(figsize=figsize)
    if len(im.squeeze().shape) == 2: plt.imshow(im, cmap='gray')
    else: plt.imshow(im)
    return plt.show()

# Cell
import pickle

def read_dict(path):
    with open(path, 'rb') as handle:
        unserialized_data = pickle.load(handle)
    return unserialized_data

def save_dict(dictionary, path):
    with open(path, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)