"""
Structured Logging Configuration for Forex Trading Platform

This module provides a centralized logging configuration that implements
structured logging with consistent formatting across all services.
"""

import json
import logging
import threading
from datetime import datetime
from typing import Any, Dict, Optional

class StructuredLogger:
    """
    StructuredLogger class.
    
    Attributes:
        Add attributes here
    """

    def __init__(
        self,
        service_name: str,
        log_level: str = "INFO",
        correlation_id: Optional[str] = None
    ):
    """
      init  .
    
    Args:
        service_name: Description of service_name
        log_level: Description of log_level
        correlation_id: Description of correlation_id
    
    """

        self.service_name = service_name
        self.correlation_id = correlation_id or threading.get_ident()
        
        # Configure the logger
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []
        
        # Add JSON formatter handler
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)

    def log(
        self,
        level: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a message with structured data.
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "correlation_id": self.correlation_id,
            "message": message,
            **(extra or {})
        }
        
        getattr(self.logger, level.lower())(
            message,
            extra={"structured": log_data}
        )

class JsonFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in JSON format.
    """
    def format(self, record: logging.LogRecord) -> str:
    """
    Format.
    
    Args:
        record: Description of record
    
    Returns:
        str: Description of return value
    
    """

        if hasattr(record, 'structured'):
            log_data = record.structured
        else:
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name
            }
        
        return json.dumps(log_data)

# Example usage:
# logger = StructuredLogger("trading-gateway")
# logger.log("INFO", "Order executed", {
#     "order_id": "123",
#     "symbol": "EUR/USD",
#     "price": 1.2345
# })
