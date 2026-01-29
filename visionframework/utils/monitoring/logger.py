"""
Logging utilities for vision framework

Enhanced logging with structured logs, context support, and flexible formatting.
"""

import logging
import sys
import json
from typing import Optional, Dict, Any, Union
from pathlib import Path
from contextvars import ContextVar

# Context storage for structured logging
_log_context: ContextVar[Dict[str, Any]] = ContextVar('log_context', default={})


class StructuredFormatter(logging.Formatter):
    """Structured log formatter that supports JSON output"""
    
    def __init__(self, format_string: str = None, json_format: bool = False):
        """
        Initialize structured formatter
        
        Args:
            format_string: Traditional log format string
            json_format: Whether to output JSON format
        """
        super().__init__(format_string)
        self.json_format = json_format
        self.format_string = format_string or "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record
        
        Args:
            record: Log record to format
        
        Returns:
            Formatted log string
        """
        if self.json_format:
            return self._format_json(record)
        else:
            return super().format(record)
    
    def _format_json(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON
        
        Args:
            record: Log record to format
        
        Returns:
            JSON formatted log string
        """
        # Base log data
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "filename": record.filename,
            "lineno": record.lineno,
            "function": record.funcName,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add context info
        context = _log_context.get()
        if context:
            log_data["context"] = context
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["timestamp", "level", "logger", "message", "module", "filename", "lineno", "function", "exc_info", "exc_text", "stack_info", "msg", "args", "created", "msecs", "relativeCreated", "thread", "threadName", "process", "processName"]:
                log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False)


class ColoredStructuredFormatter(StructuredFormatter):
    """Colored structured log formatter"""
    
    COLORS = {
        logging.DEBUG: '\033[90m',  # Gray
        logging.INFO: '\033[92m',   # Green
        logging.WARNING: '\033[93m',  # Yellow
        logging.ERROR: '\033[91m',    # Red
        logging.CRITICAL: '\033[95m'  # Purple
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors
        
        Args:
            record: Log record to format
        
        Returns:
            Formatted log string
        """
        if self.json_format:
            return super().format(record)
        
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        record.name = f"{color}{record.name}{self.RESET}"
        
        formatted = super().format(record)
        return formatted


def setup_logger(
    name: str = "visionframework",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    enable_color: bool = True,
    json_format: bool = False,
    log_config: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Setup logger for vision framework
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        format_string: Optional custom format string
        enable_color: Enable colored logging for console
        json_format: Enable JSON formatted logs
        log_config: Optional logging configuration dict
        
    Returns:
        Configured logger instance
    """
    # Apply config if provided
    if log_config:
        name = log_config.get("name", name)
        level = log_config.get("level", level)
        log_file = log_config.get("log_file", log_file)
        format_string = log_config.get("format_string", format_string)
        enable_color = log_config.get("enable_color", enable_color)
        json_format = log_config.get("json_format", json_format)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Default format with more context
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    
    # Console handler with optional color
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if enable_color:
        console_formatter = ColoredStructuredFormatter(format_string, json_format)
    else:
        console_formatter = StructuredFormatter(format_string, json_format)
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = StructuredFormatter(format_string, json_format)
        file_handler.setFormatter(file_formatter)
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


def with_log_context(**kwargs) -> Dict[str, Any]:
    """
    Create a log context dictionary
    
    Args:
        **kwargs: Context key-value pairs
        
    Returns:
        Context dictionary
    """
    return kwargs


def set_log_context(**kwargs) -> None:
    """
    Set log context for structured logging
    
    Args:
        **kwargs: Context key-value pairs
    """
    current_context = _log_context.get()
    current_context.update(kwargs)
    _log_context.set(current_context)


def clear_log_context() -> None:
    """
    Clear log context
    """
    _log_context.set({})


def update_log_context(**kwargs) -> None:
    """
    Update log context with additional key-value pairs
    
    Args:
        **kwargs: Additional context key-value pairs
    """
    current_context = _log_context.get()
    current_context.update(kwargs)
    _log_context.set(current_context)


class LogContext:
    """
    Context manager for log context
    
    Example:
        with LogContext(model="yolov8n", device="cuda"):
            logger.info("Processing video")
            # Logs will include context: {"model": "yolov8n", "device": "cuda"}
    """
    
    def __init__(self, **kwargs):
        """
        Initialize log context
        
        Args:
            **kwargs: Context key-value pairs
        """
        self.context = kwargs
        self.original_context = {}
    
    def __enter__(self):
        """
        Enter context - save original context and set new context
        """
        self.original_context = _log_context.get().copy()
        update_log_context(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context - restore original context
        """
        _log_context.set(self.original_context)


def structured_log(
    level: int,
    message: str,
    **kwargs
) -> None:
    """
    Log structured message with additional fields
    
    Args:
        level: Log level
        message: Log message
        **kwargs: Additional fields to include in log
    """
    logger = get_logger()
    extra = kwargs
    logger.log(level, message, extra=extra)


def debug_structured(message: str, **kwargs) -> None:
    """
    Log debug structured message
    
    Args:
        message: Log message
        **kwargs: Additional fields to include in log
    """
    structured_log(logging.DEBUG, message, **kwargs)


def info_structured(message: str, **kwargs) -> None:
    """
    Log info structured message
    
    Args:
        message: Log message
        **kwargs: Additional fields to include in log
    """
    structured_log(logging.INFO, message, **kwargs)


def warning_structured(message: str, **kwargs) -> None:
    """
    Log warning structured message
    
    Args:
        message: Log message
        **kwargs: Additional fields to include in log
    """
    structured_log(logging.WARNING, message, **kwargs)


def error_structured(message: str, **kwargs) -> None:
    """
    Log error structured message
    
    Args:
        message: Log message
        **kwargs: Additional fields to include in log
    """
    structured_log(logging.ERROR, message, **kwargs)


def critical_structured(message: str, **kwargs) -> None:
    """
    Log critical structured message
    
    Args:
        message: Log message
        **kwargs: Additional fields to include in log
    """
    structured_log(logging.CRITICAL, message, **kwargs)


# Add these methods to the logging.Logger class for convenience
def _logger_debug_structured(self, message: str, **kwargs):
    """Debug structured message"""
    structured_log(logging.DEBUG, message, **kwargs)

def _logger_info_structured(self, message: str, **kwargs):
    """Info structured message"""
    structured_log(logging.INFO, message, **kwargs)

def _logger_warning_structured(self, message: str, **kwargs):
    """Warning structured message"""
    structured_log(logging.WARNING, message, **kwargs)

def _logger_error_structured(self, message: str, **kwargs):
    """Error structured message"""
    structured_log(logging.ERROR, message, **kwargs)

def _logger_critical_structured(self, message: str, **kwargs):
    """Critical structured message"""
    structured_log(logging.CRITICAL, message, **kwargs)

# Monkey patch logging.Logger to add structured logging methods
logging.Logger.debug_structured = _logger_debug_structured
logging.Logger.info_structured = _logger_info_structured
logging.Logger.warning_structured = _logger_warning_structured
logging.Logger.error_structured = _logger_error_structured
logging.Logger.critical_structured = _logger_critical_structured

