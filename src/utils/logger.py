import logging
import os
import csv
from datetime import datetime

# Configure base paths
DATA_DIR = "data"

def setup_logging():
    """Initializes the centralized logging system."""
    # Ensure data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Root logger configuration
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if setup is called twice
    if not logger.handlers:
        # Console Handler (captured by crontab)
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        c_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

    return logger

# Initialize on import
logger = setup_logging()
