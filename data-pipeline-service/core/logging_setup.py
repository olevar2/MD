#!/usr/bin/env python3
"""
Structured logging setup for the service.
"""

import logging
import json
import sys
import os
import socket
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union

# Constants
SERVICE_NAME = os.environ.get("SERVICE_NAME", "unknown-service")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

class StructuredLogFormatter(logging.Formatter):
    """
    Formatter for structured JSON logs.
    """
    
    def __init__(self, service_name: str, environment: str):
        """
        Initialize the formatter.
        
        Args:
            service_name: Name of the service
            environment: Deployment environment
        """
        super().__init__()
        self.service_name = service_name
        self.environment = environment
        self.hostname = socket.gethostname()
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON string
        """
        # Basic log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "environment": self.environment,
            "hostname": self.hostname,
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "path": record.pathname,
            "line": record.lineno,
            "function": record.funcName
        }
        
        # Add correlation ID if available
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id
        
        # Add request ID if available
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        # Add user ID if available
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        
        # Add extra fields
        if hasattr(record, "data") and record.data:
            log_data["data"] = record.data
        
        # Add exception info if available
        if record.exc_info:
            exception_type, exception_value, exception_traceback = record.exc_info
            log_data["exception"] = {
                "type": exception_type.__name__,
                "message": str(exception_value),
                "traceback": traceback.format_exception(
                    exception_type, exception_value, exception_traceback
                )
            }
        
        return json.dumps(log_data)

def setup_logging(
    service_name: str = SERVICE_NAME,
    environment: str = ENVIRONMENT,
    log_level: str = LOG_LEVEL
) -> None:
    """
    Set up structured logging.
    
    Args:
        service_name: Name of the service
        environment: Deployment environment
        log_level: Log level
    """
    # Create formatter
    formatter = StructuredLogFormatter(service_name, environment)
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper()))
