"""
Logging Module

This module provides logging functionality for the platform.
"""

import logging
import json
import traceback
import sys
import os
import socket
from typing import Dict, Any, Optional, List, Callable, ClassVar
from datetime import datetime
from functools import wraps

from common_lib.config.config_manager import ConfigManager


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for logs.
    
    This formatter formats logs as JSON objects.
    """
    
    def __init__(
        self,
        service_name: str,
        include_timestamp: bool = True,
        include_hostname: bool = True,
        include_pid: bool = True,
        include_level: bool = True,
        include_logger_name: bool = True,
        include_stack_trace: bool = True,
        additional_fields: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the JSON formatter.
        
        Args:
            service_name: Name of the service
            include_timestamp: Whether to include timestamp in logs
            include_hostname: Whether to include hostname in logs
            include_pid: Whether to include process ID in logs
            include_level: Whether to include log level in logs
            include_logger_name: Whether to include logger name in logs
            include_stack_trace: Whether to include stack trace in logs
            additional_fields: Additional fields to include in logs
        """
        super().__init__()
        self.service_name = service_name
        self.include_timestamp = include_timestamp
        self.include_hostname = include_hostname
        self.include_pid = include_pid
        self.include_level = include_level
        self.include_logger_name = include_logger_name
        self.include_stack_trace = include_stack_trace
        self.additional_fields = additional_fields or {}
        
        # Get hostname
        self.hostname = socket.gethostname()
        
        # Get process ID
        self.pid = os.getpid()
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON string
        """
        # Create log object
        log = {
            "message": record.getMessage(),
            "service": self.service_name
        }
        
        # Add timestamp
        if self.include_timestamp:
            log["timestamp"] = datetime.utcfromtimestamp(record.created).isoformat() + "Z"
        
        # Add hostname
        if self.include_hostname:
            log["hostname"] = self.hostname
        
        # Add process ID
        if self.include_pid:
            log["pid"] = self.pid
        
        # Add log level
        if self.include_level:
            log["level"] = record.levelname
        
        # Add logger name
        if self.include_logger_name:
            log["logger"] = record.name
        
        # Add stack trace for errors
        if self.include_stack_trace and record.exc_info:
            log["stack_trace"] = "".join(traceback.format_exception(*record.exc_info))
        
        # Add additional fields
        for key, value in self.additional_fields.items():
            log[key] = value
        
        # Add record attributes
        for key, value in record.__dict__.items():
            if key not in ["args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName", "id", "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name", "pathname", "process", "processName", "relativeCreated", "stack_info", "thread", "threadName"]:
                log[key] = value
        
        # Convert to JSON
        return json.dumps(log)


class LoggingManager:
    """
    Logging manager for the platform.
    
    This class provides a singleton manager for logging.
    """
    
    _instance: ClassVar[Optional["LoggingManager"]] = None
    
    def __new__(cls, *args, **kwargs):
        """
        Create a new instance of the logging manager.
        
        Returns:
            Singleton instance of the logging manager
        """
        if cls._instance is None:
            cls._instance = super(LoggingManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        service_name: Optional[str] = None,
        config_manager: Optional[ConfigManager] = None
    ):
        """
        Initialize the logging manager.
        
        Args:
            service_name: Name of the service
            config_manager: Configuration manager
        """
        # Skip initialization if already initialized
        if getattr(self, "_initialized", False):
            return
        
        self.config_manager = config_manager or ConfigManager()
        
        # Get logging configuration
        try:
            logging_config = self.config_manager.get_logging_config()
            self.level = getattr(logging, logging_config.level)
            self.format = logging_config.format
            self.file = logging_config.file
            self.max_size = logging_config.max_size
            self.backup_count = logging_config.backup_count
        except Exception as e:
            # Use default values
            self.level = logging.INFO
            self.format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            self.file = None
            self.max_size = 10 * 1024 * 1024
            self.backup_count = 5
        
        # Get service name
        self.service_name = service_name
        if not self.service_name:
            try:
                app_config = self.config_manager.get_app_config()
                self.service_name = app_config.service_name
            except Exception:
                self.service_name = "unknown"
        
        # Configure logging
        self.configure_logging()
        
        self._initialized = True
    
    def configure_logging(self):
        """
        Configure logging.
        """
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.level)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        
        # Create formatter
        formatter = JsonFormatter(self.service_name)
        console_handler.setFormatter(formatter)
        
        # Add console handler to root logger
        root_logger.addHandler(console_handler)
        
        # Create file handler if file is specified
        if self.file:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.file), exist_ok=True)
                
                # Create file handler
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    self.file,
                    maxBytes=self.max_size,
                    backupCount=self.backup_count
                )
                file_handler.setLevel(self.level)
                file_handler.setFormatter(formatter)
                
                # Add file handler to root logger
                root_logger.addHandler(file_handler)
            except Exception as e:
                root_logger.error(f"Error creating file handler: {str(e)}")
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger.
        
        Args:
            name: Name of the logger
            
        Returns:
            Logger
        """
        return logging.getLogger(name)


def log_execution(logger: Optional[logging.Logger] = None, level: int = logging.INFO):
    """
    Decorator for logging function execution.
    
    Args:
        logger: Logger to use (if None, uses the function's module logger)
        level: Log level
        
    Returns:
        Decorated function
    """
    def decorator(func):
    """
    Decorator.
    
    Args:
        func: Description of func
    
    """

        @wraps(func)
        def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

            # Get logger
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            # Log function call
            logger.log(
                level,
                f"Executing {func.__name__}",
                extra={"function": func.__name__}
            )
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log function return
                logger.log(
                    level,
                    f"Completed {func.__name__}",
                    extra={"function": func.__name__}
                )
                
                return result
            except Exception as e:
                # Log function error
                logger.error(
                    f"Error in {func.__name__}: {str(e)}",
                    exc_info=True,
                    extra={"function": func.__name__}
                )
                
                # Re-raise exception
                raise
        
        return wrapper
    
    return decorator