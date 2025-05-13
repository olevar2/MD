"""
Standardized Logging Setup Module for ML Workbench Service

This module provides a standardized logging setup that follows the
common-lib pattern for logging configuration.
"""

import logging
import logging.config
import os
import sys
from typing import Dict, Any, Optional
import json
from pathlib import Path
import time
import uuid
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

from common_lib.logging import JsonFormatter, CorrelationIdFilter, RequestIdFilter
from ml_workbench_service.config.standardized_config import get_logging_config, settings

# Create logger
logger = logging.getLogger("ml_workbench_service")

def configure_logging(
    service_name: str = "ml_workbench_service",
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_json_logging: bool = True,
    enable_correlation_id: bool = True,
    enable_request_id: bool = True,
) -> None:
    """
    Configure logging for the application.

    Args:
        service_name: Name of the service
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Logging format string
        log_file: Log file path
        enable_json_logging: Whether to enable JSON logging
        enable_correlation_id: Whether to enable correlation ID
        enable_request_id: Whether to enable request ID
    """
    # Get logging config from settings
    logging_config = get_logging_config()
    
    # Override with provided parameters if any
    if log_level:
        logging_config["level"] = log_level
    if log_format:
        logging_config["format"] = log_format
    if log_file:
        logging_config["file"] = log_file
    
    # Set up basic configuration
    log_level_value = getattr(logging, logging_config["level"])
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_value)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level_value)
    
    # Create formatter
    if enable_json_logging:
        formatter = JsonFormatter(
            service_name=service_name,
            include_timestamp=True,
            include_hostname=True,
            include_level=True,
            include_logger_name=True,
        )
    else:
        formatter = logging.Formatter(logging_config["format"])
    
    console_handler.setFormatter(formatter)
    
    # Add filters if enabled
    if enable_correlation_id:
        correlation_filter = CorrelationIdFilter()
        console_handler.addFilter(correlation_filter)
    
    if enable_request_id:
        request_filter = RequestIdFilter()
        console_handler.addFilter(request_filter)
    
    # Add console handler to root logger
    root_logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if logging_config["file"]:
        log_dir = os.path.dirname(logging_config["file"])
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        if logging_config["rotation"]:
            file_handler = RotatingFileHandler(
                logging_config["file"],
                maxBytes=logging_config["max_size"],
                backupCount=logging_config["backup_count"],
            )
        else:
            file_handler = logging.FileHandler(logging_config["file"])
        
        file_handler.setLevel(log_level_value)
        file_handler.setFormatter(formatter)
        
        if enable_correlation_id:
            file_handler.addFilter(correlation_filter)
        
        if enable_request_id:
            file_handler.addFilter(request_filter)
        
        root_logger.addHandler(file_handler)
    
    # Set up library loggers
    for lib_logger_name, lib_level in [
        ("urllib3", "WARNING"),
        ("requests", "WARNING"),
        ("sqlalchemy", "WARNING"),
        ("alembic", "WARNING"),
        ("fastapi", "INFO"),
        ("uvicorn", "INFO"),
        ("mlflow", "INFO"),
    ]:
        lib_logger = logging.getLogger(lib_logger_name)
        lib_logger.setLevel(getattr(logging, lib_level))
    
    # Log startup message
    logger.info(
        f"Logging configured for {service_name}",
        extra={
            "service": service_name,
            "log_level": logging_config["level"],
            "log_file": logging_config["file"],
            "json_logging": enable_json_logging,
        },
    )

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set correlation ID for the current context.

    Args:
        correlation_id: Correlation ID to set (generates a new one if None)

    Returns:
        The correlation ID
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    CorrelationIdFilter.set_correlation_id(correlation_id)
    return correlation_id

def get_correlation_id() -> Optional[str]:
    """
    Get the current correlation ID.

    Returns:
        The current correlation ID or None if not set
    """
    return CorrelationIdFilter.get_correlation_id()

def clear_correlation_id() -> None:
    """Clear the current correlation ID."""
    CorrelationIdFilter.clear_correlation_id()

def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set request ID for the current context.

    Args:
        request_id: Request ID to set (generates a new one if None)

    Returns:
        The request ID
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    RequestIdFilter.set_request_id(request_id)
    return request_id

def get_request_id() -> Optional[str]:
    """
    Get the current request ID.

    Returns:
        The current request ID or None if not set
    """
    return RequestIdFilter.get_request_id()

def clear_request_id() -> None:
    """Clear the current request ID."""
    RequestIdFilter.clear_request_id()

def log_exception(
    logger: logging.Logger,
    exc: Exception,
    message: str = "An exception occurred",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log an exception with additional context.

    Args:
        logger: Logger instance
        exc: Exception to log
        message: Message to log
        extra: Additional context to log
    """
    if extra is None:
        extra = {}
    
    extra.update({
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    })
    
    logger.exception(message, extra=extra)

# Initialize logging with default configuration
configure_logging(
    service_name="ml_workbench_service",
    enable_json_logging=True,
    enable_correlation_id=True,
    enable_request_id=True,
)