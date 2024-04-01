from backbone import *
from core import *
from repblock import *
from config import GLOBAL_CONFIG
import torch


import logging  
import logging.config
logging.config.fileConfig("logging.conf")
logger = logging.getLogger("test")


# logger.debug(str(GLOBAL_CONFIG))


# # 
# with open("GLOBAL_CONFIG.txt","w") as file:
#     file.write(str(GLOBAL_CONFIG))

module = getattr(GLOBAL_CONFIG['channel_attention']['_pymodule'],'channel_attention')
model = module(
        ch_in = 5,
        ratio=4
    )
x = torch.ones([2,5,100,100])
out = model(x)
logger.debug(out
             )