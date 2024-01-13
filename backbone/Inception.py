import torch
import torch.nn as nn
from torchvision.models import inception_v3
from config import register
from core import get_activation,ConvNormLayer

__all__ = ['inception']

@register
class inception(nn.Module):
    standard_model = inception_v3(weights = "IMAGENET1K_V1")
    def __init__(self):
        super().__init__()
        net = self.standard_model()
    def forward(self,x):
        x = self.net(x)
        return x
        
        