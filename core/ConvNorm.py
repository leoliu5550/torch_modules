import torch
import torch.nn as nn
from .activation import get_activation

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
    
class ConvTranspose(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=0,output_padding=0, bias=False, act=None):
        super().__init__()
        self.Transconv = nn.ConvTranspose2d(
            in_channels = ch_in,
            out_channels = ch_out,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding, 
            output_padding = output_padding,
            bias = bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 
    def forward(self,x):
        return self.act(self.norm(self.Transconv(x)))
# torch.nn.ConvTranspose2d