import os
import logging
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M')}.log"
LOG_FILE_PATH = os.path.join(os.getcwd(),"logs",LOG_FILE)

logging.basicConfig(
    filename= LOG_FILE_PATH,
    format= '[ %(asctime)s ] "%(filename)s" %(name)s line: [%(lineno)d] - %(levelname)s - %(message)s',
    level = logging.INFO
)
        