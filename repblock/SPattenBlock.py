import torch
import torch.nn as nn
import torch.nn.functional as F 
from .activation import get_activation
from config import register
import logging  
import logging.config
logging.config.fileConfig("logging.conf")
logger = logging.getLogger("block.repblock")


__all__ = ['SPattenBlock']


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

# Spatail Attention Module
@register
class SPattenBlock(nn.Module):
    def __init__(self,ch_in,ch_out,act = 'relu'):
        super().__init__()
        self.ch_in = ch_in//2
        self.ch_out = ch_out//2
        self.conv1 = ConvNormLayer(self.ch_in, self.ch_out , 3, 1, padding=1, act=act)
        self.subconv2 =  ConvNormLayer(self.ch_in, self.ch_out , 3, 1, padding=1, act=act)
        self.subconv3 =  ConvNormLayer(self.ch_in, self.ch_out , 3, 1, padding=1, act=act)
        
        self.act = nn.Identity() if act is None else get_activation(act) 
        self.pooling = nn.AvgPool2d(3, 1 ,1)
    def forward(self,x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            upper_x = x[...,:self.ch_in,:,:]
            lower_x = x[...,self.ch_in:,:,:]
            upper_x = self.subconv2(self.conv1(upper_x))
            lower_x = self.subconv3(self.conv1(lower_x))
            x = torch.cat((upper_x,lower_x ),1)
            y = self.pooling(x)
            y = self.act(y)
        return y

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1


    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
