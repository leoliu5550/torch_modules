from backbone import *
from config import GLOBAL_CONFIG
from lib import summary
import torch
import logging  
import logging.config
logging.config.fileConfig("logging.conf")
logger = logging.getLogger("test")


# logger.debug(str(GLOBAL_CONFIG))

cfg = GLOBAL_CONFIG['inception']
cls = getattr(cfg['_pymodule'],'inception')
model = cls()
with open("test2.txt","w") as file:
    file.write(str(model))
    
summary(model,input_size = (3,640,640),device = torch.device("cpu"))