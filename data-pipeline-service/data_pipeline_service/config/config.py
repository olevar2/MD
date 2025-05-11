"""
Configuration Module

This module provides configuration management for the Data Pipeline Service.
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
class DataPipelineServiceConfig(ServiceSpecificConfig):
    """
    Service-specific configuration for the Data Pipeline Service.
    
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
    kafka_consumer_group_prefix: str = Field("data-pipeline", description="Prefix for Kafka consumer groups")
    kafka_auto_create_topics: bool = Field(True, description="Whether to automatically create Kafka topics")
    kafka_producer_acks: str = Field("all", description="Kafka producer acknowledgment setting")
    
    # Object storage configuration
    use_object_storage: bool = Field(False, description="Whether to use object storage")
    object_storage_endpoint: Optional[str] = Field(None, description="S3 endpoint")
    object_storage_key: Optional[str] = Field(None, description="S3 access key")
    object_storage_secret: Optional[str] = Field(None, description="S3 secret key")
    object_storage_bucket: Optional[str] = Field(None, description="S3 bucket name")
    
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


# Create a singleton ConfigManager instance
config_manager = ConfigManager(
    config_path=os.environ.get("CONFIG_PATH", "config/config.yaml"),
    service_specific_model=DataPipelineServiceConfig,
    env_prefix=os.environ.get("CONFIG_ENV_PREFIX", "DATA_PIPELINE_"),
    default_config_path=os.environ.get("DEFAULT_CONFIG_PATH", "data_pipeline_service/config/default/config.yaml")
)


# Helper functions to access configuration
def get_service_config() -> DataPipelineServiceConfig:
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
