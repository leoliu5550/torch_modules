from config import register
import torch
import torch.nn as nn
import torch.nn.functional as F 
from core import get_activation

# region 
import logging  
import logging.config
logging.config.fileConfig("logging.conf")
logger = logging.getLogger("block.cbam")
# endregion
__all__=[
    'SEnet_block'
    ]


@register
class SEnet_block(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接下降通道的倍数
    def __init__(self, ch_in, ratio=4):
        # 继承父类初始化方法
        super().__init__()
        
        # 属性分配
        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # 第一个全连接层将特征图的通道数下降4倍
        self.fc1 = nn.Linear(in_features=ch_in, out_features=ch_in//ratio, bias=False)
        nn.init.kaiming_normal_(self.fc1.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
        # relu激活
        self.relu = nn.ReLU()
        # 第二个全连接层恢复通道数
        self.fc2 = nn.Linear(in_features=ch_in//ratio, out_features=ch_in, bias=False)
        nn.init.kaiming_normal_(self.fc2.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
        # sigmoid激活函数，将权值归一化到0-1
        self.sigmoid = nn.Sigmoid()
        
    # 前向传播
    def forward(self, x):  # inputs 代表输入特征图
        ori_x = x

        # 获取输入特征图的shape
        b, c, _, _ = x.shape
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(x)
        # 维度调整 [b,c,1,1]==>[b,c]
        x = x.view([b,c])
        
        # 第一个全连接下降通道 [b,c]==>[b,c//4]
        x = self.fc1(x)
        x = self.relu(x)
        # 第二个全连接上升通道 [b,c//4]==>[b,c]
        x = self.fc2(x)
        # 对通道权重归一化处理
        x = self.sigmoid(x)
        
        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b,c,1,1])
        
        # 将输入特征图和通道权重相乘
        out = x * ori_x
        return out
