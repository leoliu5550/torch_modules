import torch
import torch.nn
import sys,pytest
sys.path.append(".")
from lib import summary
# from torchsummary import summary as summ

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
        
    @pytest.mark.skip("torchsummary having problem")
    def test_mem(self):
        # logger.debug(self.model)3
        summary(self.model,device=torch.device('cpu'),input_size=(3, 640, 640))
        # a= summary(self.model, device=torch.device('cpu'),input_size=(3, 640, 640),pr = True)["estimated_total_size"]
        # logger.debug(mem)
        assert 1== 1000        
        
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
        
    def test_mem(self):
        mem_params = sum([param.nelement()*param.element_size() for param in self.model.parameters()])
        mem_bufs = sum([buf.nelement()*buf.element_size() for buf in self.model.buffers()])
        mem = (mem_params + mem_bufs)/(1024*1024*1024)
        logger.debug(f"Total model usage {mem} gb")
        assert mem < 1000        
        
class Test_inception:
    x = torch.ones([2,3,640,640])
    cfg = GLOBAL_CONFIG['inception']
    model = getattr(cfg['_pymodule'],'inception')()
    
    def test_outputShape(self):
        out = self.model(self.x)
        assert out['feat1'].shape == torch.Size([2,256,80,80])
        assert out['feat2'].shape == torch.Size([2,256,40,40])
        assert out['feat3'].shape == torch.Size([2,256,20,20])
        
    def test_outputscale(self):
        x = torch.ones([2,3,320,320])
        out = self.model(x)
        assert out['feat1'].shape == torch.Size([2,256,40,40])
        assert out['feat2'].shape == torch.Size([2,256,20,20])
        assert out['feat3'].shape == torch.Size([2,256,10,10])
        