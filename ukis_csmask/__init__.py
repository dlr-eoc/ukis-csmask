import logging
import os
import warnings

__version__ = "0.0.1"

# disable tensorflow debugging logs
warnings.simplefilter('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True
