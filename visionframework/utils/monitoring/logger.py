"""
Logging utilities for vision framework
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def setup_logger(
    name: str = "visionframework",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    enable_color: bool = True
) -> logging.Logger:
    """
    Setup logger for vision framework
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        format_string: Optional custom format string
        enable_color: Enable colored logging for console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Default format with more context
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler with optional color
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if enable_color:
        # Add color to console output
        class ColoredFormatter(logging.Formatter):
            """Colored log formatter"""
            COLORS = {
                logging.DEBUG: '\033[90m',  # Gray
                logging.INFO: '\033[92m',   # Green
                logging.WARNING: '\033[93m',  # Yellow
                logging.ERROR: '\033[91m',    # Red
                logging.CRITICAL: '\033[95m'  # Purple
            }
            RESET = '\033[0m'
            
            def format(self, record):
                color = self.COLORS.get(record.levelno, self.RESET)
                record.levelname = f"{color}{record.levelname}{self.RESET}"
                record.name = f"{color}{record.name}{self.RESET}"
                return super().format(record)
        
        console_formatter = ColoredFormatter(format_string)
        console_handler.setFormatter(console_formatter)
    else:
        console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance
    
    Args:
        name: Logger name (default: 'visionframework')
        
    Returns:
        Logger instance
    """
    if name is None:
        name = "visionframework"
    
    logger = logging.getLogger(name)
    
    # If logger has no handlers, setup default logger
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger

