"""
Standardized Logging Setup Module for ML Integration Service

This module provides logging setup for the service using the standardized
logging configuration system from common-lib.
"""

import logging
import os
import sys
from typing import Optional, Dict, Any

from common_lib.monitoring.logging_config import (
    configure_logging,
    get_logger,
    log_with_context,
    CorrelationIdFilter
)

from config.standardized_config_1 import settings


def setup_logging(
    service_name: Optional[str] = None,
    log_level: Optional[str] = None,
    json_format: bool = True,
    correlation_id: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> None:
    """
    Set up logging for the service.
    
    Args:
        service_name: Name of the service (defaults to settings.SERVICE_NAME)
        log_level: Logging level (defaults to settings.LOG_LEVEL)
        json_format: Whether to use JSON format
        correlation_id: Correlation ID
        log_file: Path to log file (defaults to settings.LOG_FILE)
        console_output: Whether to output logs to console
    """
    # Use defaults from settings if not provided
    service_name = service_name or settings.SERVICE_NAME
    log_level = log_level or settings.LOG_LEVEL
    log_file = log_file or settings.LOG_FILE
    
    # Configure logging
    configure_logging(
        service_name=service_name,
        log_level=log_level,
        json_format=json_format,
        correlation_id=correlation_id,
        log_file=log_file,
        console_output=console_output
    )
    
    # Log startup message
    logger = get_logger(service_name)
    logger.info(
        f"Logging initialized for {service_name}",
        extra={
            "service": service_name,
            "log_level": log_level,
            "json_format": json_format,
            "log_file": log_file
        }
    )


def get_service_logger(
    name: Optional[str] = None,
    correlation_id: Optional[str] = None
) -> logging.Logger:
    """
    Get a logger for the service.
    
    Args:
        name: Logger name (defaults to service name)
        correlation_id: Correlation ID
        
    Returns:
        Logger instance
    """
    # Use service name as default logger name
    name = name or settings.SERVICE_NAME
    
    # Get logger
    return get_logger(name, correlation_id=correlation_id)


def log_model_operation(
    logger: logging.Logger,
    operation: str,
    model_name: str,
    model_version: Optional[str] = None,
    duration: Optional[float] = None,
    status: str = "success",
    error: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
) -> None:
    """
    Log a model operation.
    
    Args:
        logger: Logger instance
        operation: Operation name (e.g., "training", "prediction", "evaluation")
        model_name: Model name
        model_version: Model version
        duration: Operation duration in seconds
        status: Operation status ("success" or "failure")
        error: Error message if status is "failure"
        metrics: Model metrics
        correlation_id: Correlation ID
    """
    # Create context
    context = {
        "operation": operation,
        "model_name": model_name,
        "status": status
    }
    
    # Add optional fields
    if model_version:
        context["model_version"] = model_version
    if duration:
        context["duration"] = duration
    if error:
        context["error"] = error
    if metrics:
        context["metrics"] = metrics
    
    # Log message
    message = f"Model {operation} {status}"
    if error:
        message += f": {error}"
    
    # Determine log level
    level = logging.ERROR if status == "failure" else logging.INFO
    
    # Log with context
    log_with_context(
        logger=logger,
        level=level,
        message=message,
        context=context,
        correlation_id=correlation_id
    )


# Initialize logging when module is imported
if os.environ.get("INITIALIZE_LOGGING", "true").lower() == "true":
    setup_logging()
