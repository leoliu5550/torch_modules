import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
from .activation import get_activation
from config import register
# region
import logging  
import logging.config
logging.config.fileConfig("logging.conf")
logger = logging.getLogger("block.deform_conv")
# endregion

__all__ = ['deform_conv3']

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

@register
class deform_conv3(nn.Module):
    def __init__(self,ch_in,kernel_size=3,stride =1,padding =None,act = 'silu'):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels = ch_in,
            out_channels = ch_in,
            kernel_size = kernel_size,
            stride = stride ,
            padding = (kernel_size-1)//2 if padding is None else padding, 
        )
        self.act = nn.Identity() if act is None else get_activation(act) 
        self.conv_offset = nn.Conv2d(
            in_channels = ch_in,
            out_channels = 2*kernel_size*kernel_size,
            kernel_size=kernel_size,
            stride = stride ,
            padding = (kernel_size-1)//2 if padding is None else padding, 
        )
        nn.init.trunc_normal_(self.conv_offset.weight)

    def forward(self, x):
        
        offset = self.conv_offset(x)
        # mask = self.sig(self.conv_mask(x)) 
        out = deform_conv2d(input=x, offset=offset, 
                                    weight=self.conv.weight, 
                                        mask=None, padding=(1, 1))
        return self.act(out)

