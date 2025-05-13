"""
Standardized Configuration Module for Data Pipeline Service

This module provides configuration management for the service using the standardized
configuration management system from common-lib.
"""

import os
from functools import lru_cache
from typing import Optional, List

from pydantic import Field, field_validator

from common_lib.config import BaseAppSettings, get_settings, get_config_manager


class DataPipelineServiceSettings(BaseAppSettings):
    """
    Data Pipeline Service-specific settings.
    
    This class extends the base application settings with service-specific configuration.
    """
    
    # Override service name
    SERVICE_NAME: str = Field("data-pipeline-service", description="Name of the service")
    
    # API configuration
    API_PREFIX: str = Field("/api/v1", description="API prefix")
    
    # Data pipeline configuration
    PIPELINE_BATCH_SIZE: int = Field(1000, description="Batch size for data pipeline processing")
    PIPELINE_MAX_WORKERS: int = Field(4, description="Maximum number of workers for data pipeline processing")
    PIPELINE_TIMEOUT: int = Field(300, description="Timeout for data pipeline processing in seconds")
    
    # Market data configuration
    MARKET_DATA_SOURCES: List[str] = Field(
        ["alpha_vantage", "yahoo_finance", "oanda"],
        description="Market data sources"
    )
    MARKET_DATA_CACHE_TTL: int = Field(
        3600,
        description="Time-to-live for market data cache in seconds"
    )
    MARKET_DATA_UPDATE_INTERVAL: int = Field(
        60,
        description="Interval for market data updates in seconds"
    )
    
    # Feature store configuration
    FEATURE_STORE_ENABLED: bool = Field(
        True,
        description="Whether to enable feature store integration"
    )
    FEATURE_STORE_URL: str = Field(
        "http://feature-store-service:8000",
        description="URL of the feature store service"
    )
    FEATURE_STORE_API_KEY: Optional[str] = Field(
        None,
        description="API key for the feature store service"
    )
    
    # Data export configuration
    EXPORT_FORMATS: List[str] = Field(
        ["csv", "json", "parquet"],
        description="Supported export formats"
    )
    EXPORT_COMPRESSION: bool = Field(
        True,
        description="Whether to compress exported data"
    )
    EXPORT_CHUNK_SIZE: int = Field(
        10000,
        description="Chunk size for data export"
    )
    
    # External API configuration
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(
        None,
        description="API key for Alpha Vantage"
    )
    YAHOO_FINANCE_API_KEY: Optional[str] = Field(
        None,
        description="API key for Yahoo Finance"
    )
    OANDA_API_KEY: Optional[str] = Field(
        None,
        description="API key for Oanda"
    )
    OANDA_ACCOUNT_ID: Optional[str] = Field(
        None,
        description="Account ID for Oanda"
    )
    
    # Add validation
    @field_validator("PIPELINE_BATCH_SIZE", "PIPELINE_MAX_WORKERS", "PIPELINE_TIMEOUT")
    def validate_positive_int(cls, v: int, info) -> int:
        """
        Validate positive integer.
        
        Args:
            v: Integer value
            info: Validation info
            
        Returns:
            Validated integer value
            
        Raises:
            ValueError: If the integer is not positive
        """
        if v < 1:
            raise ValueError(f"{info.field_name} must be at least 1")
        return v
    
    @field_validator("MARKET_DATA_SOURCES")
    def validate_market_data_sources(cls, v: List[str]) -> List[str]:
        """
        Validate market data sources.
        
        Args:
            v: Market data sources
            
        Returns:
            Validated market data sources
            
        Raises:
            ValueError: If the market data sources are invalid
        """
        valid_sources = ["alpha_vantage", "yahoo_finance", "oanda", "fxcm", "dukascopy"]
        for source in v:
            if source not in valid_sources:
                raise ValueError(f"Invalid market data source: {source}. Valid sources: {valid_sources}")
        return v
    
    @field_validator("EXPORT_FORMATS")
    def validate_export_formats(cls, v: List[str]) -> List[str]:
        """
        Validate export formats.
        
        Args:
            v: Export formats
            
        Returns:
            Validated export formats
            
        Raises:
            ValueError: If the export formats are invalid
        """
        valid_formats = ["csv", "json", "parquet", "excel", "hdf5"]
        for format in v:
            if format not in valid_formats:
                raise ValueError(f"Invalid export format: {format}. Valid formats: {valid_formats}")
        return v


@lru_cache()
def get_service_settings() -> DataPipelineServiceSettings:
    """
    Get cached service settings.
    
    Returns:
        Service settings
    """
    return get_settings(
        settings_class=DataPipelineServiceSettings,
        env_file=os.environ.get("ENV_FILE", ".env"),
        config_file=os.environ.get("CONFIG_FILE", "config/config.yaml"),
        env_prefix=os.environ.get("ENV_PREFIX", "DATA_PIPELINE_")
    )


@lru_cache()
def get_service_config_manager():
    """
    Get cached service configuration manager.
    
    Returns:
        Service configuration manager
    """
    return get_config_manager(
        settings_class=DataPipelineServiceSettings,
        env_file=os.environ.get("ENV_FILE", ".env"),
        config_file=os.environ.get("CONFIG_FILE", "config/config.yaml"),
        env_prefix=os.environ.get("ENV_PREFIX", "DATA_PIPELINE_")
    )


# Create a settings instance for easy access
settings = get_service_settings()
