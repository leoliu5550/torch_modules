import torch
import torch.nn
import sys
sys.path.append(".")
from backbone import efficientnet
from config import GLOBAL_CONFIG
import logging  
import logging.config
logging.config.fileConfig("logging.conf")
logger = logging.getLogger("test")

class Test_efficientnet:
    x = torch.ones([2,3,640,640])
    cfg = GLOBAL_CONFIG['efficientnet']
    cls = getattr(cfg['_pymodule'],'efficientnet')
    model = cls()
    
    def test_outputShape(self):
        out = self.model(self.x)
        assert self.model(self.x)['feat1'].shape == torch.Size([2,256,80,80])
        assert self.model(self.x)['feat2'].shape == torch.Size([2,256,40,40])
        assert self.model(self.x)['feat3'].shape == torch.Size([2,256,20,20])