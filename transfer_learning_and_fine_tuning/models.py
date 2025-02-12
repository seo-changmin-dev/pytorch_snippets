import torch
import torch.nn as nn
from torchvision import models, transforms

class AntBeeClassifier(nn.Module):
    def __init__(self,):
        super(AntBeeClassifier, self).__init__()
        
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1, progress=True)

        self.vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=1)

    def forward(self, x):
        x = self.vgg16(x)
        return x






























































































