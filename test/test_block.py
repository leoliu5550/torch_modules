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