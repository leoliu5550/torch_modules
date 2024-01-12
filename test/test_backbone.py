import torch
import torch.nn
import sys
sys.path.append(".")
from backbone import *
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
        
class Test_yolo:
    x = torch.ones([2,3,640,640])
    cfg = GLOBAL_CONFIG['yolov1']
    cls = getattr(cfg['_pymodule'],'yolov1')
    model = cls()
    
    def test_outputShape(self):
        out = self.model(self.x)
        assert out['feat1'].shape == torch.Size([2,256,80,80])
        assert out['feat2'].shape == torch.Size([2,256,40,40])
        assert out['feat3'].shape == torch.Size([2,256,20,20])