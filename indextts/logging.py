import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging(name: str = "indextts", log_level: int | str = DEFAULT_LOG_LEVEL, log_dir: str = "logs"):
    """
    Sets up the logging configuration for the application.

    Args:
        name (str): The name of the root logger to configure.
        log_level (int | str): The logging level (e.g., logging.INFO, "DEBUG").
        log_dir (str): The directory where log files will be stored.
    """
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), DEFAULT_LOG_LEVEL)

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")

    # Get the logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Rotating)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Level: {logging.getLevelName(log_level)}, Log file: {log_file}")
    return logger

def get_logger(name: str = "indextts"):
    """
    Returns a configured logger. If not set up, it defaults to a basic configuration.
    
    Args:
        name (str): The name of the logger to retrieve.
    """
    logger = logging.getLogger(name)
    # If the logger (or its parents) hasn't been configured, standard logging might not output anything
    # if basicConfig hasn't been called. However, we rely on setup_logging being called at entry points.
    return logger
