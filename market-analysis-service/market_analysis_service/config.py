"""
Configuration module for Market Analysis Service.

This module provides configuration settings for the Market Analysis Service.
"""
import os
from typing import Dict, List, Any, Optional

# Use a simple dictionary for settings instead of pydantic to avoid dependency issues
class Settings:
    """
    Settings for the Market Analysis Service.
    """
    def __init__(self):
        # Service settings
        self.SERVICE_NAME = "market-analysis-service"
        self.VERSION = "1.0.0"
        self.DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

        # API settings
        self.API_PREFIX = "/api/v1"

        # Server settings
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", "8000"))

        # External service URLs
        self.DATA_PIPELINE_SERVICE_URL = os.getenv("DATA_PIPELINE_SERVICE_URL", "http://data-pipeline-service:8000")
        self.ANALYSIS_COORDINATOR_SERVICE_URL = os.getenv("ANALYSIS_COORDINATOR_SERVICE_URL", "http://analysis-coordinator-service:8000")
        self.FEATURE_STORE_SERVICE_URL = os.getenv("FEATURE_STORE_SERVICE_URL", "http://feature-store-service:8000")

        # Data directory
        self.DATA_DIR = os.getenv("DATA_DIR", "/data/market-analysis")

        # Logging settings
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

        # Timeout settings
        self.DEFAULT_TIMEOUT = float(os.getenv("DEFAULT_TIMEOUT", "30.0"))

        # Resilience settings
        self.RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
        self.RETRY_BACKOFF_FACTOR = float(os.getenv("RETRY_BACKOFF_FACTOR", "1.5"))
        self.CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
        self.CIRCUIT_BREAKER_RECOVERY_TIMEOUT = int(os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "30"))

settings = Settings()