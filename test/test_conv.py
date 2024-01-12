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


class Test_Convblock:
    x = torch.ones([2,30,640,640])
    cfg = GLOBAL_CONFIG['Conv2d_cdiff']
    cls = getattr(cfg['_pymodule'],'Conv2d_cdiff')
    
    def test_conv_diff(self):
        block = self.cls(
            ch_in = 30,
            ch_out = 256,
            kernel_size = 3,
            stride = 1
        )
        out = block(self.x)
        assert out.shape == torch.Size([2,256,640,640])
