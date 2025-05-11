"""
Configuration for the reconciliation framework in the ML Integration Service.

This module provides configuration settings for the reconciliation framework,
including default values for tolerance, strategies, timeouts, and retry settings.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum

from common_lib.data_reconciliation import (
    ReconciliationStrategy,
    ReconciliationSeverity,
)


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""
    
    max_retries: int = Field(3, description="Maximum number of retry attempts")
    initial_backoff_seconds: float = Field(1.0, description="Initial backoff time in seconds")
    max_backoff_seconds: float = Field(60.0, description="Maximum backoff time in seconds")
    backoff_factor: float = Field(2.0, description="Factor to multiply backoff time by on each retry")
    
    @validator('max_retries')
    def validate_max_retries(cls, v):
        """Validate max_retries value."""
        if v < 0:
            raise ValueError("max_retries must be non-negative")
        return v
    
    @validator('initial_backoff_seconds', 'max_backoff_seconds')
    def validate_backoff_times(cls, v):
        """Validate backoff time values."""
        if v <= 0:
            raise ValueError("Backoff times must be positive")
        return v
    
    @validator('backoff_factor')
    def validate_backoff_factor(cls, v):
        """Validate backoff factor value."""
        if v <= 1.0:
            raise ValueError("backoff_factor must be greater than 1.0")
        return v


class TimeoutConfig(BaseModel):
    """Configuration for timeouts."""
    
    connect_timeout_seconds: float = Field(5.0, description="Timeout for connecting to data sources")
    read_timeout_seconds: float = Field(30.0, description="Timeout for reading data from sources")
    reconciliation_timeout_seconds: float = Field(300.0, description="Timeout for the entire reconciliation process")
    
    @validator('connect_timeout_seconds', 'read_timeout_seconds', 'reconciliation_timeout_seconds')
    def validate_timeouts(cls, v):
        """Validate timeout values."""
        if v <= 0:
            raise ValueError("Timeout values must be positive")
        return v


class DataSourceConfig(BaseModel):
    """Configuration for data sources."""
    
    database_connection_string: str = Field("postgresql://user:password@localhost:5432/ml_models", description="Connection string for the database")
    cache_url: str = Field("redis://localhost:6379/0", description="URL for the cache")
    api_base_url: str = Field("http://localhost:8000/api/v1", description="Base URL for the API")
    api_timeout_seconds: float = Field(10.0, description="Timeout for API requests")
    
    @validator('api_timeout_seconds')
    def validate_api_timeout(cls, v):
        """Validate API timeout value."""
        if v <= 0:
            raise ValueError("API timeout must be positive")
        return v


class ReconciliationDefaultsConfig(BaseModel):
    """Default configuration for reconciliation processes."""
    
    default_strategy: ReconciliationStrategy = Field(ReconciliationStrategy.SOURCE_PRIORITY, description="Default strategy for resolving discrepancies")
    default_tolerance: float = Field(0.0001, description="Default tolerance for numerical differences")
    default_auto_resolve: bool = Field(True, description="Default setting for automatically resolving discrepancies")
    default_notification_threshold: ReconciliationSeverity = Field(ReconciliationSeverity.HIGH, description="Default minimum severity for notifications")
    
    @validator('default_tolerance')
    def validate_tolerance(cls, v):
        """Validate tolerance value."""
        if v < 0:
            raise ValueError("Tolerance must be non-negative")
        return v


class ReconciliationConfig(BaseModel):
    """Configuration for the reconciliation framework."""
    
    retry: RetryConfig = Field(default_factory=RetryConfig, description="Configuration for retry behavior")
    timeout: TimeoutConfig = Field(default_factory=TimeoutConfig, description="Configuration for timeouts")
    data_source: DataSourceConfig = Field(default_factory=DataSourceConfig, description="Configuration for data sources")
    defaults: ReconciliationDefaultsConfig = Field(default_factory=ReconciliationDefaultsConfig, description="Default configuration for reconciliation processes")
    
    # Maximum number of records to process in a single batch
    max_batch_size: int = Field(1000, description="Maximum number of records to process in a single batch")
    
    # Maximum number of concurrent reconciliation processes
    max_concurrent_processes: int = Field(5, description="Maximum number of concurrent reconciliation processes")
    
    # Whether to enable metrics collection
    enable_metrics: bool = Field(True, description="Whether to enable metrics collection")
    
    # Whether to enable detailed logging
    enable_detailed_logging: bool = Field(True, description="Whether to enable detailed logging")
    
    @validator('max_batch_size', 'max_concurrent_processes')
    def validate_max_values(cls, v):
        """Validate maximum values."""
        if v <= 0:
            raise ValueError("Maximum values must be positive")
        return v


# Create a default configuration instance
default_config = ReconciliationConfig()


def get_reconciliation_config() -> ReconciliationConfig:
    """
    Get the reconciliation configuration.
    
    Returns:
        Reconciliation configuration
    """
    return default_config
