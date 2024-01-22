import torch
import torch.nn
import sys
sys.path.append(".")
from repblock import *
from config import GLOBAL_CONFIG
import logging  
import logging.config
logging.config.fileConfig("logging.conf")
logger = logging.getLogger("test")

class Test_repblock:
    x = torch.ones([2,10,640,640])
    cfg = GLOBAL_CONFIG

    def test_SPattenBlock(self):
        cls = getattr(self.cfg['SPattenBlock']['_pymodule'],'SPattenBlock')
        model = cls(
            ch_in = 10,
            ch_out = 10
        )
        out = model(self.x)
        assert out.shape == torch.Size([2,10,640,640])
        
    def test_deform_conv(self):
        cls = getattr(self.cfg['deform_conv3']['_pymodule'],'deform_conv3')
        model = cls(
            ch_in = 10,
            kernel_size = 3
        )
        out = model(self.x)
        assert out.shape == torch.Size([2,10,640,640])

class TestConvattblock:
    cfg = GLOBAL_CONFIG
    def test_spatial_soft_attention(self):
        cls = getattr(self.cfg['spatial_soft_attention']['_pymodule'],'spatial_soft_attention')
        model = cls()
        x = torch.ones([2,5,100,100])
        out = model(x)
        assert out.shape == torch.Size([2,5,100,100])
        
    def test_spatial_attention(self):
        cls = getattr(self.cfg['spatial_attention']['_pymodule'],'spatial_attention')
        model = cls()
        x = torch.ones([2,5,100,100])
        out = model(x)
        assert out.shape == torch.Size([2,5,100,100])
        
    def test_channel_attention(self):
        cls = getattr(self.cfg['channel_attention']['_pymodule'],'channel_attention')
        model = cls(
            ch_in = 5,
            ratio=4
        )
        x = torch.ones([2,5,100,100])
        out = model(x)
        assert out.shape == torch.Size([2,5,100,100])
        
    def test_cbam(self):
        cls = getattr(self.cfg['cbam']['_pymodule'],'cbam')
        model = cls(
            ch_in = 10 ,ratio=4, kernel_size=3,stride =1,padding =None,act = 'silu'
        )
        x = torch.ones([4,10,8,8])
        out = model(x)
        assert out.shape == torch.Size([4,10,8,8])