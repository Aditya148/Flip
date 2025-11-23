"""Logging utilities for Flip SDK."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class FlipLogger:
    """Custom logger for Flip SDK."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern for logger."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize logger."""
        if not FlipLogger._initialized:
            self.logger = logging.getLogger("flip")
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False
            FlipLogger._initialized = True
    
    def set_level(self, level: str):
        """
        Set logging level.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        self.logger.setLevel(level_map.get(level.upper(), logging.INFO))
    
    def add_console_handler(self, level: str = "INFO", format_string: Optional[str] = None):
        """
        Add console handler.
        
        Args:
            level: Logging level
            format_string: Custom format string
        """
        # Remove existing console handlers
        self.logger.handlers = [h for h in self.logger.handlers if not isinstance(h, logging.StreamHandler)]
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper()))
        
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
    
    def add_file_handler(
        self,
        log_file: Path,
        level: str = "DEBUG",
        format_string: Optional[str] = None
    ):
        """
        Add file handler.
        
        Args:
            log_file: Path to log file
            level: Logging level
            format_string: Custom format string
        """
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_file)
        handler.setLevel(getattr(logging, level.upper()))
        
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, **kwargs)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    console: bool = True,
    format_string: Optional[str] = None
) -> FlipLogger:
    """
    Setup logging for Flip SDK.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console: Whether to log to console
        format_string: Custom format string
        
    Returns:
        FlipLogger instance
    """
    logger = FlipLogger()
    logger.set_level(level)
    
    # Clear existing handlers
    logger.logger.handlers.clear()
    
    if console:
        logger.add_console_handler(level, format_string)
    
    if log_file:
        logger.add_file_handler(log_file, "DEBUG", format_string)
    
    return logger


# Module-level convenience functions
_logger = FlipLogger()

def debug(message: str, **kwargs):
    """Log debug message."""
    _logger.debug(message, **kwargs)

def info(message: str, **kwargs):
    """Log info message."""
    _logger.info(message, **kwargs)

def warning(message: str, **kwargs):
    """Log warning message."""
    _logger.warning(message, **kwargs)

def error(message: str, **kwargs):
    """Log error message."""
    _logger.error(message, **kwargs)

def critical(message: str, **kwargs):
    """Log critical message."""
    _logger.critical(message, **kwargs)

def exception(message: str, **kwargs):
    """Log exception with traceback."""
    _logger.exception(message, **kwargs)
