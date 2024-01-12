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
    cfg = GLOBAL_CONFIG['SPattenBlock']
    cls = getattr(cfg['_pymodule'],'SPattenBlock')
    def test_SPattenBlock(self):
        model = self.cls(
            ch_in = 10,
            ch_out = 10
        )
        out = model(self.x)
        assert out.shape == torch.Size([2,10,640,640])
        
        