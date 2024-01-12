from torchvision.models import efficientnet_v2_s
import torch
import torch.nn as nn
from config import register

__all__ = ['efficientnet']

class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding=None, **kwargs):
        super(CNNBlock,self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size, 
            padding=(kernel_size-1)//2 if padding is None else padding,bias=False,
            **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self,x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x

@register
class efficientnet(nn.Module):
    standard_model = efficientnet_v2_s(weights="IMAGENET1K_V1")
    def __init__(self):
        super().__init__()
        self.bs3 = nn.Sequential(
            self.standard_model.features[0],
            # self.standard_model.features[1],
            self.standard_model.features[2],
            self.standard_model.features[3],
        )
        # self.bs3 = self.standard_model.features[0:4]
        # [-1, 96, 80, 80]
        self.bs4 = nn.Sequential(
            self.standard_model.features[4],
            self.standard_model.features[5][0:2]
        )
        # self.bs4 = self.standard_model.features[4:6]# [-1, 224, 40, 40]
        self.bs5 = nn.Sequential(
            self.standard_model.features[6][0:2],
            self.standard_model.features[7]
        )
        # self.bs5 = self.standard_model.features[6:]# [-1, 1280, 20, 20]
        
        self.input_proj = nn.ModuleList()
        for in_channel in [64,160,1280]: # 1280
            self.input_proj.append(
                CNNBlock(in_channels=in_channel,out_channels=256,kernel_size=3)
            )
        self._reset_parameters()
        # 64, 80, 80
        # 160, 40, 40
        # 256, 20, 20
    def _reset_parameters(self):
        for submodel in [self.bs3,self.bs4,self.bs5]:
            for layers in submodel:
                for lay in layers:
                    if lay== "Conv2d":
                        nn.init.normal_(lay.weight)
                        
    def forward(self,x):
        feat1 = self.bs3(x)
        feat2 = self.bs4(feat1)
        feat3 = self.bs5(feat2)
        
        return {"feat1":self.input_proj[0](feat1),
                "feat2":self.input_proj[1](feat2),
                "feat3":self.input_proj[2](feat3)}

        # for proj in self.input_proj:
        #     nn.init.normal_(proj.weight)