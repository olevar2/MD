"""
Structured Logger Module.

Provides a standardized logging functionality with JSON formatting
and context propagation for use across all services.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional


class StructuredLogger:
    """
    Structured logger with JSON output and context propagation.
    """

    def __init__(
        self,
        service_name: str,
        log_level: str = "INFO",
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the structured logger.

        Args:
            service_name: Name of the service using the logger
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            context: Optional dictionary with context information to include in logs
        """
        self.service_name = service_name
        self.context = context or {}
        
        # Setup logger
        self.logger = logging.getLogger(service_name)
        level = getattr(logging, log_level.upper())
        self.logger.setLevel(level)
        
        # Add console handler with JSON formatting
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(self._json_formatter)
        self.logger.addHandler(handler)
    
    def _json_formatter(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": self.service_name,
            "level": record.levelname,
            "message": record.getMessage(),
            **self.context
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self._format_traceback(record.exc_info[2])
            }
            
        # Add extra attributes from record
        if hasattr(record, "extra"):
            log_data.update(record.extra)
            
        return json.dumps(log_data)
    
    def _format_traceback(self, tb):
        """Format traceback to string."""
        import traceback
        if tb:
            return traceback.format_tb(tb)
        return None
    
    def with_context(self, **kwargs) -> 'StructuredLogger':
        """
        Create a new logger instance with additional context.
        
        Args:
            **kwargs: Key-value pairs to add to the context
            
        Returns:
            A new logger instance with the updated context
        """
        new_context = {**self.context, **kwargs}
        return StructuredLogger(self.service_name, log_level=self.logger.level, context=new_context)
    
    def debug(self, msg: str, **kwargs):
        """Log debug message with optional context."""
        self._log(logging.DEBUG, msg, **kwargs)
        
    def info(self, msg: str, **kwargs):
        """Log info message with optional context."""
        self._log(logging.INFO, msg, **kwargs)
        
    def warning(self, msg: str, **kwargs):
        """Log warning message with optional context."""
        self._log(logging.WARNING, msg, **kwargs)
        
    def error(self, msg: str, **kwargs):
        """Log error message with optional context."""
        self._log(logging.ERROR, msg, **kwargs)
        
    def critical(self, msg: str, **kwargs):
        """Log critical message with optional context."""
        self._log(logging.CRITICAL, msg, **kwargs)
    
    def _log(self, level: int, msg: str, **kwargs):
        """Internal method to log with proper context."""
        extra = kwargs.pop("extra", {})
        if kwargs:
            extra["extra"] = kwargs
        self.logger.log(level, msg, extra=extra)


def get_logger(service_name: str, log_level: str = "INFO") -> StructuredLogger:
    """
    Helper function to get a configured logger.
    
    Args:
        service_name: Name of the service using the logger
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured StructuredLogger instance
    """
    return StructuredLogger(service_name, log_level)