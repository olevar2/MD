"""
Mock implementation of common_lib for testing.
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
import os
import json
import logging

# Mock AppSettings
class AppSettings(BaseModel):
    """Base settings class for all services."""

    # API settings
    HOST: str = "localhost"
    PORT: int = 8000
    DEBUG_MODE: bool = False
    LOG_LEVEL: str = "INFO"

    # Security settings
    JWT_SECRET: str = "test_secret"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION: int = 3600

    # Database settings
    DATABASE_URL_OVERRIDE: Optional[str] = None
    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 5432
    DATABASE_NAME: str = "testdb"
    DATABASE_USER: str = "test"
    DATABASE_PASSWORD: str = "test"

    # Redis settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    # Service URLs
    MARKET_DATA_SERVICE_URL: str = "http://localhost:8001"
    NOTIFICATION_SERVICE_URL: str = "http://localhost:8002"

    # Monitoring settings
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9000

    # WebSocket settings
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_MAX_CONNECTIONS: int = 1000

    @property
    def DATABASE_URL(self) -> str:
        """Get the database URL."""
        if self.DATABASE_URL_OVERRIDE:
            return self.DATABASE_URL_OVERRIDE

        return f"postgresql://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"

    @property
    def REDIS_URL(self) -> str:
        """Get the Redis URL."""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

# Mock load_settings function
def load_settings(settings_class, env_file: Optional[str] = None) -> Any:
    """Load settings from environment variables or .env file."""
    return settings_class()

# Mock exceptions
class BaseError(Exception):
    """Base error class for all exceptions."""

    def __init__(self, message: str, code: str = "UNKNOWN_ERROR", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

class ValidationError(BaseError):
    """Validation error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", details)

class NotFoundError(BaseError):
    """Resource not found error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "NOT_FOUND", details)

class ServiceUnavailableError(BaseError):
    """Service unavailable error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "SERVICE_UNAVAILABLE", details)

class DatabaseError(BaseError):
    """Database error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DATABASE_ERROR", details)

class AuthenticationError(BaseError):
    """Authentication error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "AUTHENTICATION_ERROR", details)

class AuthorizationError(BaseError):
    """Authorization error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "AUTHORIZATION_ERROR", details)

class RateLimitError(BaseError):
    """Rate limit error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "RATE_LIMIT_ERROR", details)

class TimeoutError(BaseError):
    """Timeout error."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "TIMEOUT_ERROR", details)

# Mock config module
class ConfigModule:
    """Mock config module."""

    AppSettings = AppSettings
    load_settings = load_settings

# Expose config module
config = ConfigModule()
