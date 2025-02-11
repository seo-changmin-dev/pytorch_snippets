import os
from glob import glob
from typing import List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

class HymenopteraDataSet(torch.utils.data.Dataset):
    '''
    [Input]
    phase: [train, val]

    [Output]
    label_data: [0, 1] # ants: 0 | bees: 1
    '''
    def __init__(self, data_path_list_dict:dict, phase:str, transform:transforms):
        super(HymenopteraDataSet, self).__init__()
        self.data_path_list = data_path_list_dict
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.data_path_list[self.phase])

    def __getitem__(self, idx):
        data_path = self.data_path_list[self.phase][idx]

        # data
        image_data = Image.open(data_path)
        image_data = self.transform(image_data)

        # label
        label_name = data_path.split('/')[-2]
        if label_name == "ants":
            label_data = 0
        elif label_name == "bees":
            label_data = 1

        return image_data, label_data

def get_data_path_list(data_dir:str):
    data_dir = './data/hymenoptera_data'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    glob_regex = f"{train_dir}/**/*.jpg"
    train_data_path_list = glob(glob_regex, recursive=True)

    glob_regex = f"{val_dir}/**/*.jpg"
    val_data_path_list = glob(glob_regex, recursive=True)

    data_path_list_dict = {
        'train': train_data_path_list,
        'val': val_data_path_list
    }

    return data_path_list_dict

if __name__ == "__main__":
    data_dir = './data'
    data_path_list_dict = get_data_path_list(data_dir)

    print(data_path_list_dict['train'].__len__(), data_path_list_dict['val'].__len__())