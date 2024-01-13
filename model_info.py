import logging  
import logging.config
logging.config.fileConfig("logging.conf")
logger = logging.getLogger("test")

from backbone import *
from repblock import *
from config import GLOBAL_CONFIG
from torchsummary import summary,kk
cfg = GLOBAL_CONFIG['efficientnet']
cls = getattr(cfg['_pymodule'],'efficientnet')
model = cls()
    


a = summary(model, input_size=(3, 640, 640))
print(a['total_size']<1500)