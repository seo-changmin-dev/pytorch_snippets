import os
from glob import glob
from typing import List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
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
            label_data = 0.0
        elif label_name == "bees":
            label_data = 1.0

        return image_data, label_data

class HymenopteraDataBuilder():
    def __init__(self, data_dir:str, transform_dict:dict, batch_size:int):
        self.data_path_list_dict = get_data_path_list(data_dir)
        self.transform_dict = transform_dict
        self.batch_size = batch_size
    
    def get_dataset(self):
        train_dataset = HymenopteraDataSet(self.data_path_list_dict, 'train', self.transform_dict['train'])
        val_dataset = HymenopteraDataSet(self.data_path_list_dict, 'val', self.transform_dict['val'])

        split_indices = np.arange(len(val_dataset))
        np.random.shuffle(split_indices)

        split_separator = len(split_indices) // 2
        val_indices = split_indices[:split_separator]
        test_indices = split_indices[split_separator+1:]

        test_dataset = torch.utils.data.Subset(val_dataset, test_indices) 
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

        dataset_dict = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

        return dataset_dict
    
    def get_dataloader(self):
        dataset_dict = self.get_dataset()

        train_dataloader = torch.utils.data.DataLoader(dataset=dataset_dict['train'], batch_size=self.batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(dataset=dataset_dict['val'], batch_size=self.batch_size, shuffle=False)
        test_dataloader = torch.utils.data.DataLoader(dataset=dataset_dict['test'], batch_size=self.batch_size, shuffle=False)

        dataloader_dict = {
            'train': train_dataloader,
            'val': val_dataloader,
            'test': test_dataloader
        }

        return dataloader_dict

class ImageTransformBuilder():
    def __init__(self, resize, mean, std):
        self.transform = {
            'train': transforms.Compose([transforms.Resize(resize),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([transforms.Resize(resize),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std)
            ]),
        }

    def __call__(self, phase:str):
        return self.transform[phase]

def get_data_path_list(data_dir:str):
    data_dir = os.path.join(data_dir, 'hymenoptera_data')
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