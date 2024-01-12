import torch
import torch.nn as nn
import torch.nn.functional as F 
from core import get_activation
import math
from config import register
import logging  
import logging.config
logging.config.fileConfig("logging.conf")
logger = logging.getLogger("block.conv_diff")

__all__ = ['Conv2d_cdiff']



@register
class Conv2d_cdiff(nn.Module):
    def __init__(self,
                ch_in, 
                ch_out, 
                kernel_size, 
                stride, 
                padding=None, 
                bias=False, 
                act='relu',
                theta=0.7):

        super().__init__() 
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        nn.init.normal_(self.conv.weight)
        self.theta = theta
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8: 
            return out_normal 
        else:
            #pdb.set_trace()
            # [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)

            kernel_diff = kernel_diff[:, :, None, None]

            out_diff = F.conv2d(
                input=x, 
                weight=kernel_diff, 
                bias=self.conv.bias, 
                stride=self.conv.stride, 
                padding=0
                )
            out = out_normal - self.theta * out_diff
            out = self.act(out)
            return out
