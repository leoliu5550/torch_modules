import logging  
import logging.config
logging.config.fileConfig("logging.conf")
logger = logging.getLogger("test")

from backbone import *
from repblock import *
from config import GLOBAL_CONFIG
from torchsummary import summary

cfg = GLOBAL_CONFIG['Conv2d_cdiff']
cls = getattr(cfg['_pymodule'],'Conv2d_cdiff')

model = cls()

summary(model, input_size=(3, 640, 640))