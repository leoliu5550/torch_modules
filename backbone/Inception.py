import torch
import torch.nn as nn
from torchvision.models import inception_v3
from config import register
from core import get_activation

__all__ = ['inception']

@register
class inception(nn.Module):
    standard_model = inception_v3(weights = "IMAGENET1K_V1")
    def __init__(self):
        super().__init__()
        
        
class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
