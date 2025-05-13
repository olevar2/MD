"""
Logging Configuration for Strategy Execution Engine

This module provides logging configuration for the Strategy Execution Engine.
"""

import os
import sys
import json
import logging
import logging.config
from datetime import datetime
from typing import Dict, Any, Optional

from config.config_1 import get_settings

class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record
            
        Returns:
            str: JSON formatted log
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in log_data:
                continue
            
            # Skip internal attributes
            if key in ("args", "asctime", "created", "exc_info", "exc_text", 
                       "filename", "funcName", "id", "levelname", "levelno", 
                       "lineno", "module", "msecs", "message", "msg", "name", 
                       "pathname", "process", "processName", "relativeCreated", 
                       "stack_info", "thread", "threadName"):
                continue
            
            log_data[key] = value
        
        return json.dumps(log_data)

def configure_logging() -> None:
    """
    Configure logging for the application.
    """
    settings = get_settings()
    
    # Determine log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # Configure logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
            "json": {
                "()": JsonFormatter
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "standard" if not settings.debug_mode else "json",
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "json",
                "filename": "logs/strategy_execution_engine.log",
                "maxBytes": 10485760,  # 10 MB
                "backupCount": 5
            }
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": True
            },
            "uvicorn": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False
            },
            "uvicorn.access": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False
            },
            "strategy_execution_engine": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": False
            }
        }
    }
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Log configuration complete
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level {settings.log_level}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)

class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter for adding context to log messages.
    """
    
    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        """
        Initialize logger adapter.
        
        Args:
            logger: Logger instance
            extra: Extra context to add to log messages
        """
        super().__init__(logger, extra or {})
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Process log message.
        
        Args:
            msg: Log message
            kwargs: Keyword arguments
            
        Returns:
            tuple: Processed message and kwargs
        """
        # Add correlation ID if available
        if "correlation_id" in self.extra:
            kwargs.setdefault("extra", {})["correlation_id"] = self.extra["correlation_id"]
        
        return msg, kwargs

def get_logger_with_context(name: str, correlation_id: Optional[str] = None, 
                           extra: Optional[Dict[str, Any]] = None) -> LoggerAdapter:
    """
    Get a logger with context.
    
    Args:
        name: Logger name
        correlation_id: Correlation ID
        extra: Extra context
        
    Returns:
        LoggerAdapter: Logger adapter with context
    """
    logger = get_logger(name)
    
    context = extra or {}
    if correlation_id:
        context["correlation_id"] = correlation_id
    
    return LoggerAdapter(logger, context)
