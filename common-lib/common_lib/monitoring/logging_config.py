"""
Standardized Logging Configuration Module

This module provides standardized logging configuration for the platform.
It supports structured logging with JSON format, correlation IDs, and
integration with distributed tracing.
"""

import json
import logging
import logging.config
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Union

import pythonjsonlogger.jsonlogger as jsonlogger

# Check if OpenTelemetry is available
try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanContext, TraceFlags
    
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False


class CorrelationIdFilter(logging.Filter):
    """
    Filter that adds correlation ID to log records.
    
    This filter adds a correlation ID to log records, which can be used to
    correlate log messages across services.
    """
    
    def __init__(self, name: str = "", correlation_id: Optional[str] = None):
        """
        Initialize the filter.
        
        Args:
            name: Filter name
            correlation_id: Correlation ID
        """
        super().__init__(name)
        self.correlation_id = correlation_id or str(uuid.uuid4())
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log record.
        
        Args:
            record: Log record
            
        Returns:
            True to include the record, False to exclude it
        """
        # Add correlation ID to record
        record.correlation_id = getattr(record, "correlation_id", self.correlation_id)
        
        # Add trace context if available
        if OPENTELEMETRY_AVAILABLE:
            span = trace.get_current_span()
            if span and span.get_span_context().is_valid:
                context = span.get_span_context()
                record.trace_id = format(context.trace_id, "032x")
                record.span_id = format(context.span_id, "016x")
                record.trace_flags = format(context.trace_flags, "02x")
        
        return True


class StructuredLogFormatter(jsonlogger.JsonFormatter):
    """
    Formatter for structured logs in JSON format.
    
    This formatter formats log records as JSON objects with additional fields.
    """
    
    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        json_default: Optional[Any] = None,
        json_encoder: Optional[Any] = None,
        json_indent: Optional[int] = None,
        json_separators: Optional[tuple] = None,
        prefix: str = "",
        rename_fields: Optional[Dict[str, str]] = None,
        timestamp_field: str = "timestamp",
        timestamp_format: str = "%Y-%m-%dT%H:%M:%S.%fZ"
    ):
        """
        Initialize the formatter.
        
        Args:
            fmt: Format string
            datefmt: Date format string
            style: Style of the format string
            json_default: Function to convert non-serializable objects to JSON
            json_encoder: JSON encoder class
            json_indent:
    """
    json_indent class.
    
    Attributes:
        Add attributes here
    """
 JSON indentation
            json_separators: JSON separators
            prefix: Prefix for log messages
            rename_fields: Dictionary of field name mappings
            timestamp_field: Name of the timestamp field
            timestamp_format: Format of the timestamp field
        """
        if fmt is None:
            fmt = "%(timestamp)s %(level)s %(name)s %(message)s"
        
        super().__init__(
            fmt=fmt,
            datefmt=datefmt,
            style=style,
            json_default=json_default,
            json_encoder=json_encoder,
            json_indent=json_indent,
            json_separators=json_separators,
            prefix=prefix,
            rename_fields=rename_fields
        )
        
        self.timestamp_field = timestamp_field
        self.timestamp_format = timestamp_format
    
    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any]
    ) -> None:
        """
        Add fields to the log record.
        
        Args:
            log_record: Log record dictionary
            record: Log record
            message_dict: Message dictionary
        """
        # Add timestamp
        log_record[self.timestamp_field] = datetime.utcnow().strftime(self.timestamp_format)
        
        # Add log level
        log_record["level"] = record.levelname
        
        # Add correlation ID
        if hasattr(record, "correlation_id"):
            log_record["correlation_id"] = record.correlation_id
        
        # Add trace context
        if hasattr(record, "trace_id"):
            log_record["trace_id"] = record.trace_id
        if hasattr(record, "span_id"):
            log_record["span_id"] = record.span_id
        if hasattr(record, "trace_flags"):
            log_record["trace_flags"] = record.trace_flags
        
        # Add extra fields
        for key, value in message_dict.items():
            log_record[key] = value
        
        # Add extra attributes
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            for key, value in record.extra.items():
                log_record[key] = value
        
        super().add_fields(log_record, record, message_dict)


def configure_logging(
    service_name: str,
    log_level: Union[str, int] = logging.INFO,
    json_format: bool = True,
    correlation_id: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> None:
    """
    Configure logging for the service.
    
    Args:
        service_name: Name of the service
        log_level: Logging level
        json_format: Whether to use JSON format
        correlation_id: Correlation ID
        log_file: Path to log file
        console_output: Whether to output logs to console
    """
    # Convert log level to integer
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create handlers
    handlers = {}
    
    # Add console handler
    if console_output:
        if json_format:
            console_formatter = StructuredLogFormatter()
        else:
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "json" if json_format else "standard",
            "stream": "ext://sys.stdout"
        }
    
    # Add file handler
    if log_file:
        if json_format:
            file_formatter = StructuredLogFormatter()
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        handlers["file"] = {
            "class": "logging.FileHandler",
            "level": log_level,
            "formatter": "json" if json_format else "standard",
            "filename": log_file
        }
    
    # Create formatters
    formatters = {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }
    
    if json_format:
        formatters["json"] = {
            "()": "common_lib.monitoring.logging_config.StructuredLogFormatter"
        }
    
    # Create filters
    filters = {
        "correlation_id": {
            "()": "common_lib.monitoring.logging_config.CorrelationIdFilter",
            "correlation_id": correlation_id
        }
    }
    
    # Create logging configuration
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "filters": filters,
        "handlers": handlers,
        "loggers": {
            "": {
                "level": log_level,
                "handlers": list(handlers.keys()),
                "filters": ["correlation_id"],
                "propagate": True
            },
            service_name: {
                "level": log_level,
                "handlers": list(handlers.keys()),
                "filters": ["correlation_id"],
                "propagate": False
            }
        }
    }
    
    # Configure logging
    logging.config.dictConfig(logging_config)


def get_logger(
    name: str,
    correlation_id: Optional[str] = None
) -> logging.Logger:
    """
    Get a logger with correlation ID.
    
    Args:
        name: Logger name
        correlation_id: Correlation ID
        
    Returns:
        Logger with correlation ID
    """
    logger = logging.getLogger(name)
    
    # Add correlation ID filter
    if correlation_id:
        for handler in logger.handlers:
            handler.addFilter(CorrelationIdFilter(correlation_id=correlation_id))
    
    return logger


def log_with_context(
    logger: logging.Logger,
    level: Union[str, int],
    message: str,
    context: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
    exc_info: Optional[Any] = None
) -> None:
    """
    Log a message with context.
    
    Args:
        logger: Logger
        level: Logging level
        message: Log message
        context: Log context
        correlation_id: Correlation ID
        exc_info: Exception info
    """
    # Convert level to integer
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Create extra dictionary
    extra = {"extra": context or {}}
    
    # Add correlation ID
    if correlation_id:
        extra["correlation_id"] = correlation_id
    
    # Log message
    logger.log(level, message, extra=extra, exc_info=exc_info)
