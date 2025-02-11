import os
from glob import glob
from typing import List

import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

class HymenopteraDataSet(torch.utils.data.Dataset):
    def __init__(self, file_list:List[str], phase:str, transform:transforms):
        super(HymenopteraDataSet, self).__init__()
        self.file_list = file_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

def get_data_path_list(data_dir:str):
    data_dir = './data/hymenoptera_data'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    glob_regex = f"{train_dir}/**/*"
    train_file_list = glob(glob_regex, recursive=True)

    glob_regex = f"{val_dir}/**/*"
    val_file_list = glob(glob_regex, recursive=True)

    file_list_dict = {
        'train': train_file_list,
        'val': val_file_list
    }

    return file_list_dict

if __name__ == "__main__":
    data_dir = './data'
    file_list_dict = get_data_path_list(data_dir)

    print(file_list_dict['train'].__len__(), file_list_dict['val'].__len__())