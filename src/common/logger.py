"""Logging configuration for the application."""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "mllifecycle",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Setup logger with console and optional file output.
    
    Args:
        name: Logger name (use root logger name for all modules)
        level: Logging level
        log_file: Optional path to log file
    
    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler - always add with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Force UTF-8 encoding for console to handle special characters
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass
    
    # Simple, clean formatter
    formatter = logging.Formatter(
        fmt='%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler - optional with UTF-8 encoding
    if log_file:
        log_path = Path(log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(level)
        
        # More detailed formatter for file
        file_formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "mllifecycle") -> logging.Logger:
    """Get logger instance. Will inherit configuration from setup_logger.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
