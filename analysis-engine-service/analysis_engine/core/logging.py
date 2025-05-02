"""
Logging Module

This module provides logging functionality for the Analysis Engine Service.
"""

import logging
import sys
from typing import Optional
from analysis_engine.config import get_settings

def configure_logging() -> None:
    """Configure logging for the application."""
    settings = get_settings()

    # Configure root logger
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("analysis_engine.log")
        ]
    )

    # Set log levels for specific modules
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: The name of the logger

    Returns:
        logging.Logger: The logger instance
    """
    return logging.getLogger(name or __name__)