import os
import random
import numpy as np

import torch
import torch.nn as nn

from dataset import HymenopteraDataBuilder, ImageTransformBuilder
from models import AntBeeClassifier

from tqdm import tqdm
from tabulate import tabulate

DATA_DIR = './data'

LR = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 32

VGG16_SIZE = (256, 256)
VGG16_MEAN = (0.485, 0.456, 0.406)
VGG16_STD = (0.229, 0.224, 0.225)

SEED = 0
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Dataloader
    transform_builder = ImageTransformBuilder(VGG16_SIZE, VGG16_MEAN, VGG16_STD)
    transform_dict = {'train': transform_builder('train'), 'val': transform_builder('val')}
    dataloader_builder = HymenopteraDataBuilder(data_dir=DATA_DIR, transform_dict=transform_dict, batch_size=BATCH_SIZE)
    dataloader_dict = dataloader_builder.get_dataloader()

    model = AntBeeClassifier().to(DEVICE)

    # Fine-tuning setting (1)
    params_to_update_1 = []
    params_to_update_2 = []
    params_to_update_3 = []

    update_param_names_1 = ["features"]
    update_param_names_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
    update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

    for name, param in model.vgg16.named_parameters():
        if update_param_names_1[0] in name: # "features" 라는 substring이 name의 일부인 경우
            param.required_grad = True
            params_to_update_1.append(param)
        elif name in update_param_names_2: # name이 update_param_names_2 리스트 안에 포함되는 경우
            param.required_grad = True
            params_to_update_2.append(param)
        elif name in update_param_names_3:
            param.required_grad = True
            params_to_update_3.append(param)
        else:
            param.required_grad = False
    
    # Fine-tuning setting (2)
    optimizer = torch.optim.SGD(params = [
        {'params': params_to_update_1, 'lr': 1e-4},
        {'params': params_to_update_2, 'lr': 5e-4},
        {'params': params_to_update_3, 'lr': 1e-3},
    ], momentum=0.9)

    criterion = nn.BCEWithLogitsLoss()

    # Train & Validation
    for epoch in range(0, NUM_EPOCHS+1):
        epoch_log = []

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            if epoch == 0 and phase =='train':
                continue

            running_loss = 0
            running_correct = 0

            for image_batch, label_batch in tqdm(dataloader_dict[phase]):
                image_batch = image_batch.to(DEVICE)
                label_batch = label_batch.to(DEVICE)

                optimizer.zero_grad()

                label_pred = model(image_batch)
                
                loss = criterion(label_pred.flatten(), label_batch)
                pred = (label_pred.flatten() > 0).float()

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * image_batch.size(0)
                running_correct += torch.sum(pred == label_batch)

            running_loss /= len(dataloader_dict[phase].dataset)
            running_acc = running_correct.float() /  len(dataloader_dict[phase].dataset)

            epoch_log.append([epoch, phase, f"{running_loss:.6f}", f"{running_acc:.4f}"])
        print(tabulate(epoch_log, headers=["Epoch", "Phase", "Loss", "Accuracy"], tablefmt="rst"))

    # Test
    epoch_log = []
    phase = 'test'

    model.eval()

    running_loss = 0
    running_correct = 0

    for image_batch, label_batch in tqdm(dataloader_dict[phase]):
        image_batch = image_batch.to(DEVICE)
        label_batch = label_batch.to(DEVICE)

        label_pred = model(image_batch)
        
        loss = criterion(label_pred.flatten(), label_batch)
        pred = (label_pred.flatten() > 0).float()

        running_loss += loss.item() * image_batch.size(0)
        running_correct += torch.sum(pred == label_batch)

    running_loss /= len(dataloader_dict[phase].dataset)
    running_acc = running_correct.float() /  len(dataloader_dict[phase].dataset)

    epoch_log.append([epoch, phase, f"{running_loss:.6f}", f"{running_acc:.4f}"])
    print(tabulate(epoch_log, headers=["Epoch", "Phase", "Loss", "Accuracy"], tablefmt="rst"))

    checkpoint_path = './results/checkpoints'
    save_path = os.path.join(checkpoint_path, f"{epoch}.pth")

    print(f"Model is saved in {save_path}")
    torch.save(model, save_path)
    