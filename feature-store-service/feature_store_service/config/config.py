"""
Configuration Module

This module provides configuration management for the Feature Store Service.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Type

from pydantic import BaseModel, Field, validator

from common_lib.config import (
    ServiceSpecificConfig,
    ConfigManager
)


# Define service-specific configuration
class FeatureStoreServiceConfig(ServiceSpecificConfig):
    """
    Service-specific configuration for the Feature Store Service.
    
    This class defines the service-specific configuration parameters.
    """
    
    # Service configuration
    api_prefix: str = Field("/api/v1", description="API prefix")
    cors_origins: List[str] = Field(["*"], description="CORS origins")
    max_workers: int = Field(4, description="Maximum number of worker threads")
    cache_size: int = Field(1000, description="Maximum number of items in the cache")
    max_requests_per_minute: int = Field(60, description="Maximum number of API requests per minute")
    max_retries: int = Field(3, description="Maximum number of retries for failed requests")
    retry_delay_seconds: int = Field(5, description="Delay between retries in seconds")
    timeout_seconds: int = Field(30, description="Timeout for API requests in seconds")
    
    # Kafka configuration
    kafka_bootstrap_servers: str = Field("localhost:9092", description="Comma-separated list of Kafka broker addresses")
    kafka_consumer_group_prefix: str = Field("feature-store", description="Prefix for Kafka consumer groups")
    kafka_auto_create_topics: bool = Field(True, description="Whether to automatically create Kafka topics")
    kafka_producer_acks: str = Field("all", description="Kafka producer acknowledgment setting")
    
    # Feature store configuration
    max_feature_age_days: int = Field(30, description="Maximum age of features in days")
    default_timeframe: str = Field("1h", description="Default timeframe for features")
    supported_timeframes: List[str] = Field(
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
        description="Supported timeframes for features"
    )
    default_indicators: List[str] = Field(
        ["sma", "ema", "rsi", "macd", "bollinger_bands"],
        description="Default indicators to compute"
    )
    enable_feature_versioning: bool = Field(True, description="Whether to enable feature versioning")
    feature_cache_ttl_seconds: int = Field(300, description="Time-to-live for feature cache in seconds")
    batch_size: int = Field(1000, description="Batch size for feature computation")
    max_parallel_computations: int = Field(4, description="Maximum number of parallel feature computations")
    
    @validator("max_workers")
    def validate_max_workers(cls, v):
        """Validate maximum number of workers."""
        if v < 1:
            raise ValueError("Maximum workers must be at least 1")
        return v
    
    @validator("cache_size")
    def validate_cache_size(cls, v):
        """Validate cache size."""
        if v < 0:
            raise ValueError("Cache size must be non-negative")
        return v
    
    @validator("max_requests_per_minute")
    def validate_max_requests_per_minute(cls, v):
        """Validate maximum number of API requests per minute."""
        if v < 1:
            raise ValueError("Maximum requests per minute must be at least 1")
        return v
    
    @validator("max_retries")
    def validate_max_retries(cls, v):
        """Validate maximum number of retries."""
        if v < 0:
            raise ValueError("Maximum retries must be non-negative")
        return v
    
    @validator("retry_delay_seconds")
    def validate_retry_delay_seconds(cls, v):
        """Validate retry delay."""
        if v < 0:
            raise ValueError("Retry delay must be non-negative")
        return v
    
    @validator("timeout_seconds")
    def validate_timeout_seconds(cls, v):
        """Validate timeout."""
        if v < 0:
            raise ValueError("Timeout must be non-negative")
        return v
    
    @validator("max_feature_age_days")
    def validate_max_feature_age_days(cls, v):
        """Validate maximum feature age."""
        if v < 1:
            raise ValueError("Maximum feature age must be at least 1 day")
        return v
    
    @validator("feature_cache_ttl_seconds")
    def validate_feature_cache_ttl_seconds(cls, v):
        """Validate feature cache TTL."""
        if v < 0:
            raise ValueError("Feature cache TTL must be non-negative")
        return v
    
    @validator("batch_size")
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if v < 1:
            raise ValueError("Batch size must be at least 1")
        return v
    
    @validator("max_parallel_computations")
    def validate_max_parallel_computations(cls, v):
        """Validate maximum number of parallel computations."""
        if v < 1:
            raise ValueError("Maximum parallel computations must be at least 1")
        return v


# Create a singleton ConfigManager instance
config_manager = ConfigManager(
    config_path=os.environ.get("CONFIG_PATH", "config/config.yaml"),
    service_specific_model=FeatureStoreServiceConfig,
    env_prefix=os.environ.get("CONFIG_ENV_PREFIX", "FEATURE_STORE_"),
    default_config_path=os.environ.get("DEFAULT_CONFIG_PATH", "feature_store_service/config/default/config.yaml")
)


# Helper functions to access configuration
def get_service_config() -> FeatureStoreServiceConfig:
    """
    Get the service-specific configuration.
    
    Returns:
        Service-specific configuration
    """
    return config_manager.get_service_specific_config()


def get_database_config():
    """
    Get the database configuration.
    
    Returns:
        Database configuration
    """
    return config_manager.get_database_config()


def get_logging_config():
    """
    Get the logging configuration.
    
    Returns:
        Logging configuration
    """
    return config_manager.get_logging_config()


def get_service_client_config(service_name: str):
    """
    Get the configuration for a specific service client.
    
    Args:
        service_name: Name of the service
        
    Returns:
        Service client configuration
    """
    return config_manager.get_service_client_config(service_name)


def get_feature_store_config():
    """
    Get the feature store configuration.
    
    Returns:
        Feature store configuration
    """
    service_config = get_service_config()
    return {
        "max_feature_age_days": service_config.max_feature_age_days,
        "default_timeframe": service_config.default_timeframe,
        "supported_timeframes": service_config.supported_timeframes,
        "default_indicators": service_config.default_indicators,
        "enable_feature_versioning": service_config.enable_feature_versioning,
        "feature_cache_ttl_seconds": service_config.feature_cache_ttl_seconds,
        "batch_size": service_config.batch_size,
        "max_parallel_computations": service_config.max_parallel_computations
    }


def is_development() -> bool:
    """
    Check if the application is running in development mode.
    
    Returns:
        True if the application is running in development mode, False otherwise
    """
    return config_manager.is_development()


def is_testing() -> bool:
    """
    Check if the application is running in testing mode.
    
    Returns:
        True if the application is running in testing mode, False otherwise
    """
    return config_manager.is_testing()


def is_production() -> bool:
    """
    Check if the application is running in production mode.
    
    Returns:
        True if the application is running in production mode, False otherwise
    """
    return config_manager.is_production()
