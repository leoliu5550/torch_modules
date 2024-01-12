from backbone import efficientnet
from config import *
import logging  
import logging.config
logging.config.fileConfig("logging.conf")
logger = logging.getLogger("test")


logger.debug(str(GLOBAL_CONFIG))
# with open("test.txt","w") as file:
#     file.write(str(GLOBAL_CONFIG))