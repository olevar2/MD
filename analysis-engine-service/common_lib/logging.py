"""
Common logging functionality for the forex trading platform.

This module provides logging utilities used across multiple services.
"""

import logging
import json
import uuid
import time
import threading
from typing import Dict, Any, Optional

# Thread-local storage for request context
_thread_local = threading.local()


def get_request_id() -> str:
    """
    Get the current request ID from thread-local storage.
    
    Returns:
        Request ID or a new UUID if not set
    """
    if not hasattr(_thread_local, "request_id"):
        _thread_local.request_id = str(uuid.uuid4())
    return _thread_local.request_id


def set_request_id(request_id: str) -> None:
    """
    Set the request ID in thread-local storage.
    
    Args:
        request_id: Request ID
    """
    _thread_local.request_id = request_id


def get_correlation_id() -> str:
    """
    Get the current correlation ID from thread-local storage.
    
    Returns:
        Correlation ID or a new UUID if not set
    """
    if not hasattr(_thread_local, "correlation_id"):
        _thread_local.correlation_id = str(uuid.uuid4())
    return _thread_local.correlation_id


def set_correlation_id(correlation_id: str) -> None:
    """
    Set the correlation ID in thread-local storage.
    
    Args:
        correlation_id: Correlation ID
    """
    _thread_local.correlation_id = correlation_id


class StructuredLogFormatter(logging.Formatter):
    """
    Formatter for structured JSON logs.
    
    This formatter outputs logs in JSON format with additional context.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.
        
        Args:
            record: Log record
            
        Returns:
            Formatted log message
        """
        # Get basic log data
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
            "correlation_id": get_correlation_id()
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ["args", "asctime", "created", "exc_info", "exc_text", "filename",
                          "funcName", "id", "levelname", "levelno", "lineno", "module",
                          "msecs", "message", "msg", "name", "pathname", "process",
                          "processName", "relativeCreated", "stack_info", "thread", "threadName"]:
                log_data[key] = value
        
        return json.dumps(log_data)


def configure_logging(
    service_name: str,
    log_level: str = "INFO",
    json_format: bool = True
) -> None:
    """
    Configure logging for a service.
    
    Args:
        service_name: Name of the service
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Whether to use JSON format for logs
    """
    # Convert log level string to constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    
    # Set formatter
    if json_format:
        formatter = StructuredLogFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s"
        )
    
    console_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(console_handler)
    
    # Create logger for the service
    logger = logging.getLogger(service_name)
    logger.setLevel(numeric_level)
    
    # Log configuration
    logger.info(f"Logging configured for service: {service_name}", extra={
        "service": service_name,
        "log_level": log_level,
        "json_format": json_format
    })


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context to log messages.
    
    This adapter automatically adds request_id and correlation_id to log messages.
    """
    
    def process(self, msg, kwargs):
        """
        Process the log message and add context.
        
        Args:
            msg: Log message
            kwargs: Additional keyword arguments
            
        Returns:
            Tuple of (message, kwargs)
        """
        # Add request_id and correlation_id to extra
        kwargs.setdefault("extra", {})
        kwargs["extra"].setdefault("request_id", get_request_id())
        kwargs["extra"].setdefault("correlation_id", get_correlation_id())
        
        return msg, kwargs


def get_logger(name: str) -> LoggerAdapter:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger adapter
    """
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, {})
