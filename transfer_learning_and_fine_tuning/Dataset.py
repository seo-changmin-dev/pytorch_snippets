import os
from glob import glob
from typing import List

import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

class HymenopteraDataSet(torch.utils.data.Dataset):
    '''
    phase: [train, val]
    '''
    def __init__(self, data_path_list_dict:dict, phase:str, transform:transforms):
        super(HymenopteraDataSet, self).__init__()
        self.data_path_list = data_path_list_dict
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.data_path_list[self.phase])

    def __getitem__(self, idx):
        # data
        
        # label
        pass

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