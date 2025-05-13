"""
Logging Module

This module provides logging functionality for the service.
"""

import logging
import sys
from typing import Optional

from common_lib.correlation import get_correlation_id


class CorrelationFilter(logging.Filter):
    """Logging filter that adds correlation ID to log records."""
    
    def filter(self, record):
    """
    Filter.
    
    Args:
        record: Description of record
    
    """

        correlation_id = get_correlation_id()
        record.correlation_id = correlation_id or "no-correlation-id"
        return True


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level
    """
    # Convert string log level to numeric value
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Set formatter with correlation ID
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s"
    )
    console_handler.setFormatter(formatter)
    
    # Add correlation filter
    correlation_filter = CorrelationFilter()
    console_handler.addFilter(correlation_filter)
    
    # Add handler to root logger
    root_logger.addHandler(console_handler)
    
    # Log configuration
    logging.info(f"Logging configured with level: {log_level}")
