"""
Configuration Schema Module

This module defines the schema for the configuration system.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator


class DatabaseConfig(BaseModel):
    """
    Database configuration.
    """
    
    host: str = Field(..., description="Database host")
    port: int = Field(..., description="Database port")
    username: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")
    database: str = Field(..., description="Database name")
    pool_size: int = Field(10, description="Connection pool size")
    max_overflow: int = Field(20, description="Maximum number of connections to overflow")
    pool_timeout: int = Field(30, description="Pool timeout in seconds")
    pool_recycle: int = Field(1800, description="Pool recycle time in seconds")
    echo: bool = Field(False, description="Echo SQL statements")
    
    @validator("port")
    def validate_port(cls, v):
        """Validate port number."""
        if v < 1 or v > 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @validator("pool_size")
    def validate_pool_size(cls, v):
        """Validate pool size."""
        if v < 1:
            raise ValueError("Pool size must be at least 1")
        return v


class LoggingConfig(BaseModel):
    """
    Logging configuration.
    """
    
    level: str = Field("INFO", description="Logging level")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Logging format"
    )
    file: Optional[str] = Field(None, description="Log file path")
    max_size: int = Field(10 * 1024 * 1024, description="Maximum log file size in bytes")
    backup_count: int = Field(5, description="Number of backup log files")
    
    @validator("level")
    def validate_level(cls, v):
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v not in valid_levels:
            raise ValueError(f"Level must be one of {valid_levels}")
        return v


class ServiceConfig(BaseModel):
    """
    Service configuration.
    """
    
    host: str = Field("0.0.0.0", description="Service host")
    port: int = Field(..., description="Service port")
    workers: int = Field(4, description="Number of workers")
    timeout: int = Field(60, description="Timeout in seconds")
    
    @validator("port")
    def validate_port(cls, v):
        """Validate port number."""
        if v < 1 or v > 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @validator("workers")
    def validate_workers(cls, v):
        """Validate number of workers."""
        if v < 1:
            raise ValueError("Workers must be at least 1")
        return v


class RetryConfig(BaseModel):
    """
    Retry configuration.
    """
    
    max_retries: int = Field(3, description="Maximum number of retries")
    initial_backoff: float = Field(1.0, description="Initial backoff time in seconds")
    max_backoff: float = Field(60.0, description="Maximum backoff time in seconds")
    backoff_factor: float = Field(2.0, description="Backoff factor")
    
    @validator("max_retries")
    def validate_max_retries(cls, v):
        """Validate maximum number of retries."""
        if v < 0:
            raise ValueError("Maximum retries must be non-negative")
        return v
    
    @validator("initial_backoff")
    def validate_initial_backoff(cls, v):
        """Validate initial backoff time."""
        if v <= 0:
            raise ValueError("Initial backoff must be positive")
        return v
    
    @validator("max_backoff")
    def validate_max_backoff(cls, v):
        """Validate maximum backoff time."""
        if v <= 0:
            raise ValueError("Maximum backoff must be positive")
        return v
    
    @validator("backoff_factor")
    def validate_backoff_factor(cls, v):
        """Validate backoff factor."""
        if v <= 1.0:
            raise ValueError("Backoff factor must be greater than 1.0")
        return v


class CircuitBreakerConfig(BaseModel):
    """
    Circuit breaker configuration.
    """
    
    failure_threshold: int = Field(5, description="Failure threshold")
    recovery_timeout: float = Field(60.0, description="Recovery timeout in seconds")
    expected_exceptions: List[str] = Field(
        ["ConnectionError", "Timeout"],
        description="Expected exceptions"
    )
    
    @validator("failure_threshold")
    def validate_failure_threshold(cls, v):
        """Validate failure threshold."""
        if v < 1:
            raise ValueError("Failure threshold must be at least 1")
        return v
    
    @validator("recovery_timeout")
    def validate_recovery_timeout(cls, v):
        """Validate recovery timeout."""
        if v <= 0:
            raise ValueError("Recovery timeout must be positive")
        return v


class ServiceClientConfig(BaseModel):
    """
    Service client configuration.
    """
    
    base_url: str = Field(..., description="Base URL for the service")
    timeout: float = Field(30.0, description="Request timeout in seconds")
    retry: RetryConfig = Field(default_factory=RetryConfig, description="Retry configuration")
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig,
        description="Circuit breaker configuration"
    )
    
    @validator("timeout")
    def validate_timeout(cls, v):
        """Validate timeout."""
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v


class ServiceClientsConfig(BaseModel):
    """
    Service clients configuration.
    """
    
    market_data_service: ServiceClientConfig = Field(
        ...,
        description="Market Data Service client configuration"
    )
    feature_store_service: ServiceClientConfig = Field(
        ...,
        description="Feature Store Service client configuration"
    )
    analysis_engine_service: ServiceClientConfig = Field(
        ...,
        description="Analysis Engine Service client configuration"
    )
    trading_service: ServiceClientConfig = Field(
        ...,
        description="Trading Service client configuration"
    )


class AppConfig(BaseModel):
    """
    Application configuration.
    """
    
    environment: str = Field("development", description="Environment")
    debug: bool = Field(False, description="Debug mode")
    testing: bool = Field(False, description="Testing mode")
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment."""
        valid_environments = ["development", "testing", "staging", "production"]
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of {valid_environments}")
        return v


class ServiceSpecificConfig(BaseModel):
    """
    Base class for service-specific configuration.
    
    This class should be extended by each service to add its specific configuration.
    """
    
    pass


class Config(BaseModel):
    """
    Main configuration class.
    """
    
    app: AppConfig = Field(..., description="Application configuration")
    database: DatabaseConfig = Field(..., description="Database configuration")
    logging: LoggingConfig = Field(..., description="Logging configuration")
    service: ServiceConfig = Field(..., description="Service configuration")
    service_clients: ServiceClientsConfig = Field(
        ...,
        description="Service clients configuration"
    )
    service_specific: Optional[ServiceSpecificConfig] = Field(
        None,
        description="Service-specific configuration"
    )