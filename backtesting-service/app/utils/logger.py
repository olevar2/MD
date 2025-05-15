"""
Logger utility for the backtesting service.
"""
import logging
import sys
from typing import Optional

from app.utils.correlation_id import get_correlation_id

class CorrelationIdFilter(logging.Filter):
    """Filter that adds correlation ID to log records."""
    
    def filter(self, record):
        """Add correlation ID to log record."""
        record.correlation_id = get_correlation_id() or "-"
        return True

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with correlation ID.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Add correlation ID filter if not already added
    for handler in logging.root.handlers:
        if not any(isinstance(f, CorrelationIdFilter) for f in handler.filters):
            handler.addFilter(CorrelationIdFilter())
    
    return logger

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(correlation_id)s] [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Add correlation ID filter to root logger
    for handler in logging.root.handlers:
        handler.addFilter(CorrelationIdFilter())
