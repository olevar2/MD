"""
Configuration Module

This module provides configuration management for the Analysis Engine Service.
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
class AnalysisEngineServiceConfig(ServiceSpecificConfig):
    """
    Service-specific configuration for the Analysis Engine Service.
    
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
    kafka_consumer_group_prefix: str = Field("analysis-engine", description="Prefix for Kafka consumer groups")
    kafka_auto_create_topics: bool = Field(True, description="Whether to automatically create Kafka topics")
    kafka_producer_acks: str = Field("all", description="Kafka producer acknowledgment setting")
    
    # Analysis engine configuration
    max_analysis_age_days: int = Field(30, description="Maximum age of analyses in days")
    default_timeframe: str = Field("1h", description="Default timeframe for analyses")
    supported_timeframes: List[str] = Field(
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
        description="Supported timeframes for analyses"
    )
    default_indicators: List[str] = Field(
        ["sma", "ema", "rsi", "macd", "bollinger_bands"],
        description="Default indicators to compute"
    )
    enable_analysis_versioning: bool = Field(True, description="Whether to enable analysis versioning")
    analysis_cache_ttl_seconds: int = Field(300, description="Time-to-live for analysis cache in seconds")
    batch_size: int = Field(1000, description="Batch size for analysis computation")
    max_parallel_analyses: int = Field(4, description="Maximum number of parallel analyses")
    
    # Machine learning configuration
    ml_model_storage_path: str = Field("models", description="Path to store ML models")
    ml_enable_gpu: bool = Field(False, description="Whether to enable GPU for ML models")
    ml_max_training_time_minutes: int = Field(60, description="Maximum training time for ML models in minutes")
    ml_validation_split: float = Field(0.2, description="Validation split for ML models")
    ml_test_split: float = Field(0.1, description="Test split for ML models")
    ml_hyperparameter_tuning: bool = Field(True, description="Whether to enable hyperparameter tuning for ML models")
    
    # Technical analysis configuration
    ta_enable_advanced_patterns: bool = Field(True, description="Whether to enable advanced pattern recognition")
    ta_pattern_recognition_threshold: float = Field(0.7, description="Threshold for pattern recognition")
    ta_fibonacci_levels: List[float] = Field(
        [0.236, 0.382, 0.5, 0.618, 0.786],
        description="Fibonacci levels for technical analysis"
    )
    
    # Sentiment analysis configuration
    sentiment_enable_sentiment_analysis: bool = Field(True, description="Whether to enable sentiment analysis")
    sentiment_sources: List[str] = Field(
        ["news", "social_media", "economic_calendar"],
        description="Sources for sentiment analysis"
    )
    sentiment_update_interval_minutes: int = Field(30, description="Update interval for sentiment analysis in minutes")
    
    # Risk analysis configuration
    risk_enable_risk_analysis: bool = Field(True, description="Whether to enable risk analysis")
    risk_var_confidence_level: float = Field(0.95, description="Confidence level for Value at Risk (VaR)")
    risk_stress_test_scenarios: List[str] = Field(
        ["market_crash", "volatility_spike", "liquidity_crisis"],
        description="Scenarios for stress testing"
    )
    
    # Correlation analysis configuration
    correlation_enable_correlation_analysis: bool = Field(True, description="Whether to enable correlation analysis")
    correlation_window_days: int = Field(30, description="Window for correlation analysis in days")
    correlation_min_correlation_threshold: float = Field(0.7, description="Minimum correlation threshold")
    
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
    
    @validator("max_analysis_age_days")
    def validate_max_analysis_age_days(cls, v):
        """Validate maximum analysis age."""
        if v < 1:
            raise ValueError("Maximum analysis age must be at least 1 day")
        return v
    
    @validator("analysis_cache_ttl_seconds")
    def validate_analysis_cache_ttl_seconds(cls, v):
        """Validate analysis cache TTL."""
        if v < 0:
            raise ValueError("Analysis cache TTL must be non-negative")
        return v
    
    @validator("batch_size")
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if v < 1:
            raise ValueError("Batch size must be at least 1")
        return v
    
    @validator("max_parallel_analyses")
    def validate_max_parallel_analyses(cls, v):
        """Validate maximum number of parallel analyses."""
        if v < 1:
            raise ValueError("Maximum parallel analyses must be at least 1")
        return v
    
    @validator("ml_max_training_time_minutes")
    def validate_ml_max_training_time_minutes(cls, v):
        """Validate maximum training time."""
        if v < 1:
            raise ValueError("Maximum training time must be at least 1 minute")
        return v
    
    @validator("ml_validation_split")
    def validate_ml_validation_split(cls, v):
        """Validate validation split."""
        if v < 0 or v > 1:
            raise ValueError("Validation split must be between 0 and 1")
        return v
    
    @validator("ml_test_split")
    def validate_ml_test_split(cls, v):
        """Validate test split."""
        if v < 0 or v > 1:
            raise ValueError("Test split must be between 0 and 1")
        return v
    
    @validator("ta_pattern_recognition_threshold")
    def validate_ta_pattern_recognition_threshold(cls, v):
        """Validate pattern recognition threshold."""
        if v < 0 or v > 1:
            raise ValueError("Pattern recognition threshold must be between 0 and 1")
        return v
    
    @validator("sentiment_update_interval_minutes")
    def validate_sentiment_update_interval_minutes(cls, v):
        """Validate sentiment update interval."""
        if v < 1:
            raise ValueError("Sentiment update interval must be at least 1 minute")
        return v
    
    @validator("risk_var_confidence_level")
    def validate_risk_var_confidence_level(cls, v):
        """Validate VaR confidence level."""
        if v < 0 or v > 1:
            raise ValueError("VaR confidence level must be between 0 and 1")
        return v
    
    @validator("correlation_window_days")
    def validate_correlation_window_days(cls, v):
        """Validate correlation window."""
        if v < 1:
            raise ValueError("Correlation window must be at least 1 day")
        return v
    
    @validator("correlation_min_correlation_threshold")
    def validate_correlation_min_correlation_threshold(cls, v):
        """Validate minimum correlation threshold."""
        if v < 0 or v > 1:
            raise ValueError("Minimum correlation threshold must be between 0 and 1")
        return v


# Create a singleton ConfigManager instance
config_manager = ConfigManager(
    config_path=os.environ.get("CONFIG_PATH", "config/config.yaml"),
    service_specific_model=AnalysisEngineServiceConfig,
    env_prefix=os.environ.get("CONFIG_ENV_PREFIX", "ANALYSIS_ENGINE_"),
    default_config_path=os.environ.get("DEFAULT_CONFIG_PATH", "analysis_engine/config/default/config.yaml")
)


# Helper functions to access configuration
def get_service_config() -> AnalysisEngineServiceConfig:
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


def get_analysis_engine_config():
    """
    Get the analysis engine configuration.
    
    Returns:
        Analysis engine configuration
    """
    service_config = get_service_config()
    return {
        "max_analysis_age_days": service_config.max_analysis_age_days,
        "default_timeframe": service_config.default_timeframe,
        "supported_timeframes": service_config.supported_timeframes,
        "default_indicators": service_config.default_indicators,
        "enable_analysis_versioning": service_config.enable_analysis_versioning,
        "analysis_cache_ttl_seconds": service_config.analysis_cache_ttl_seconds,
        "batch_size": service_config.batch_size,
        "max_parallel_analyses": service_config.max_parallel_analyses,
        "ml": {
            "model_storage_path": service_config.ml_model_storage_path,
            "enable_gpu": service_config.ml_enable_gpu,
            "max_training_time_minutes": service_config.ml_max_training_time_minutes,
            "validation_split": service_config.ml_validation_split,
            "test_split": service_config.ml_test_split,
            "hyperparameter_tuning": service_config.ml_hyperparameter_tuning
        },
        "ta": {
            "enable_advanced_patterns": service_config.ta_enable_advanced_patterns,
            "pattern_recognition_threshold": service_config.ta_pattern_recognition_threshold,
            "fibonacci_levels": service_config.ta_fibonacci_levels
        },
        "sentiment": {
            "enable_sentiment_analysis": service_config.sentiment_enable_sentiment_analysis,
            "sources": service_config.sentiment_sources,
            "update_interval_minutes": service_config.sentiment_update_interval_minutes
        },
        "risk": {
            "enable_risk_analysis": service_config.risk_enable_risk_analysis,
            "var_confidence_level": service_config.risk_var_confidence_level,
            "stress_test_scenarios": service_config.risk_stress_test_scenarios
        },
        "correlation": {
            "enable_correlation_analysis": service_config.correlation_enable_correlation_analysis,
            "window_days": service_config.correlation_window_days,
            "min_correlation_threshold": service_config.correlation_min_correlation_threshold
        }
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
