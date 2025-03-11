"""
Logging utility for the application.
"""

import logging
import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from config import settings

def setup_logger():
    """
    Set up and configure the application logger.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("localllm_qa")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Create handlers
    file_handler = logging.FileHandler(settings.LOG_FILE)
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Set formatter for handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create and export logger instance
logger = setup_logger()