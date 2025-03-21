import os
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    FORMATS = {
        logging.DEBUG: "\033[36m%(asctime)s - %(name)s - %(levelname)s - %(message)s\033[0m",
        logging.INFO: "\033[32m%(asctime)s - %(name)s - %(levelname)s - %(message)s\033[0m",
        logging.WARNING: "\033[33m%(asctime)s - %(name)s - %(levelname)s - %(message)s\033[0m",
        logging.ERROR: "\033[31m%(asctime)s - %(name)s - %(levelname)s - %(message)s\033[0m",
        logging.CRITICAL: "\033[1;31m%(asctime)s - %(name)s - %(levelname)s - %(message)s\033[0m",
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def setup_logging(name="goalkickltd_trading_bot"):
    """Set up logging configuration"""
    level = LOG_LEVELS.get(LOG_LEVEL, logging.INFO)
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(CustomFormatter())
    
    # File handler for all logs
    file_handler = RotatingFileHandler(
        logs_dir / "trading_bot.log", 
        maxBytes=10485760,  # 10MB
        backupCount=20
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    
    # File handler for errors only
    error_handler = RotatingFileHandler(
        logs_dir / "errors.log", 
        maxBytes=10485760,  # 10MB
        backupCount=20
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    
    # Time-based rotating handler for daily logs
    daily_handler = TimedRotatingFileHandler(
        logs_dir / "daily.log",
        when="midnight",
        interval=1,
        backupCount=30
    )
    daily_handler.setLevel(level)
    daily_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(daily_handler)
    
    return logger

def get_logger(name):
    """Get logger for a specific module"""
    return logging.getLogger(f"goalkickltd_trading_bot.{name}")