# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/61_textbox_dataset_sroie2019.ipynb (unless otherwise specified).

__all__ = ['sroie_textbox_config', 'read_data', 'get_filename2bboxes_dict', 'create_df']

# Cell
from .core import save_dict, read_dict, plot
from fastai import *
from fastai.vision import *
import pandas as pd
import numpy as np
import cv2
from tqdm.notebook import tqdm

# Cell
class sroie_textbox_config:
    MAIN_DIR = '../data/sroie2019/'
    FILE_DIR = '../data/textbox/sroie2019bbox.pickle'

# Cell
def read_data(csv_path='images/X00016469670.txt'):
    ''' returns [([4,2], str),...] '''
    out = []
    with open(csv_path, encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
            if len(line) > 8:
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, line[:8]))
                label = ','.join(line[8:])
                points = np.array([ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ])
                out.append([points, label])
    return out

# Cell
def get_filename2bboxes_dict():
    return read_dict(sroie_textbox_config.FILE_DIR)

# Cell
def create_df():
    data = []
    for mode in ['train', 'test']:
        path = os.path.join(sroie_textbox_config.MAIN_DIR, mode + '_img')
        for fn in os.listdir(path):
            data.append((os.path.join(path, fn), mode == 'test', 'sroie2019'))
    return pd.DataFrame(data, columns=['image_path', 'valid', 'dataset'])