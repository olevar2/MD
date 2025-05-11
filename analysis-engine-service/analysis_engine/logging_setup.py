"""
Logging Setup Module

This module provides logging setup for the Analysis Engine Service.
"""

import os
import logging
import logging.config
from typing import Dict, Any, Optional

from common_lib.config import LoggingConfig
from analysis_engine.config import get_logging_config


def setup_logging(service_name: str = "analysis-engine-service", logging_config: Optional[LoggingConfig] = None) -> logging.Logger:
    """
    Set up logging for the service.
    
    Args:
        service_name: Name of the service
        logging_config: Logging configuration (if None, uses the configuration from the config manager)
        
    Returns:
        Logger instance
    """
    # Get logging configuration
    if logging_config is None:
        logging_config = get_logging_config()
    
    # Set up logging
    log_level = getattr(logging, logging_config.level.upper())
    log_format = logging_config.format
    log_file = logging_config.file
    
    # Create logger
    logger = logging.getLogger(service_name)
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger
