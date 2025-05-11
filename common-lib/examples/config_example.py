"""
Configuration Example

This example demonstrates how to use the configuration management system in the common-lib package.
"""

import os
import logging
import tempfile
import yaml
from typing import Dict, Any, Optional, List

from common_lib.config import (
    Config,
    AppConfig,
    DatabaseConfig,
    LoggingConfig,
    ServiceConfig,
    RetryConfig,
    CircuitBreakerConfig,
    ServiceClientConfig,
    ServiceClientsConfig,
    ServiceSpecificConfig,
    ConfigLoader,
    ConfigManager
)
from pydantic import BaseModel, Field, validator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("config-example")


# Define a service-specific configuration model
class AnalysisEngineConfig(ServiceSpecificConfig):
    """Analysis Engine Service specific configuration."""
    
    max_workers: int = Field(4, description="Maximum number of worker threads")
    cache_size: int = Field(1000, description="Maximum number of items in the cache")
    default_timeframe: str = Field("1h", description="Default timeframe for analysis")
    supported_timeframes: List[str] = Field(
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        description="Supported timeframes"
    )
    
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
    
    @validator("default_timeframe")
    def validate_default_timeframe(cls, v, values):
        """Validate default timeframe."""
        if "supported_timeframes" in values and v not in values["supported_timeframes"]:
            raise ValueError(f"Default timeframe must be one of {values['supported_timeframes']}")
        return v


# Create a sample configuration
def create_sample_config() -> Dict[str, Any]:
    """Create a sample configuration dictionary."""
    return {
        "app": {
            "environment": "development",
            "debug": True,
            "testing": False
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "username": "postgres",
            "password": "password",
            "database": "forex_platform"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": None
        },
        "service": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 4,
            "timeout": 60
        },
        "service_clients": {
            "market_data_service": {
                "base_url": "http://market-data-service:8001",
                "timeout": 30.0,
                "retry": {
                    "max_retries": 3,
                    "initial_backoff": 1.0,
                    "max_backoff": 60.0,
                    "backoff_factor": 2.0
                },
                "circuit_breaker": {
                    "failure_threshold": 5,
                    "recovery_timeout": 60.0,
                    "expected_exceptions": ["ConnectionError", "Timeout"]
                }
            },
            "feature_store_service": {
                "base_url": "http://feature-store-service:8002",
                "timeout": 30.0
            },
            "analysis_engine_service": {
                "base_url": "http://analysis-engine-service:8003",
                "timeout": 30.0
            },
            "trading_service": {
                "base_url": "http://trading-service:8004",
                "timeout": 30.0
            }
        },
        "service_specific": {
            "max_workers": 8,
            "cache_size": 2000,
            "default_timeframe": "1h",
            "supported_timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        }
    }


# Example 1: Load configuration from a file
def example_load_from_file():
    """Example of loading configuration from a file."""
    logger.info("Example 1: Load configuration from a file")
    
    # Create a temporary configuration file
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        config_path = f.name
        yaml.dump(create_sample_config(), f)
    
    try:
        # Load configuration
        config_loader = ConfigLoader(logger=logger)
        config = config_loader.load_config(
            config_path=config_path,
            service_specific_model=AnalysisEngineConfig
        )
        
        # Print configuration
        logger.info(f"App environment: {config.app.environment}")
        logger.info(f"Database host: {config.database.host}")
        logger.info(f"Service port: {config.service.port}")
        logger.info(f"Market Data Service URL: {config.service_clients.market_data_service.base_url}")
        logger.info(f"Service-specific max workers: {config.service_specific.max_workers}")
    finally:
        # Clean up
        os.unlink(config_path)


# Example 2: Override configuration with environment variables
def example_override_with_env_vars():
    """Example of overriding configuration with environment variables."""
    logger.info("\nExample 2: Override configuration with environment variables")
    
    # Create a temporary configuration file
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        config_path = f.name
        yaml.dump(create_sample_config(), f)
    
    try:
        # Set environment variables
        os.environ["APP_APP__ENVIRONMENT"] = "production"
        os.environ["APP_DATABASE__HOST"] = "db.example.com"
        os.environ["APP_SERVICE__PORT"] = "9000"
        os.environ["APP_SERVICE_SPECIFIC__MAX_WORKERS"] = "16"
        
        # Load configuration
        config_loader = ConfigLoader(logger=logger)
        config = config_loader.load_config(
            config_path=config_path,
            service_specific_model=AnalysisEngineConfig,
            env_prefix="APP_"
        )
        
        # Print configuration
        logger.info(f"App environment: {config.app.environment}")
        logger.info(f"Database host: {config.database.host}")
        logger.info(f"Service port: {config.service.port}")
        logger.info(f"Service-specific max workers: {config.service_specific.max_workers}")
    finally:
        # Clean up
        os.unlink(config_path)
        del os.environ["APP_APP__ENVIRONMENT"]
        del os.environ["APP_DATABASE__HOST"]
        del os.environ["APP_SERVICE__PORT"]
        del os.environ["APP_SERVICE_SPECIFIC__MAX_WORKERS"]


# Example 3: Use the ConfigManager
def example_config_manager():
    """Example of using the ConfigManager."""
    logger.info("\nExample 3: Use the ConfigManager")
    
    # Create a temporary configuration file
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        config_path = f.name
        yaml.dump(create_sample_config(), f)
    
    try:
        # Initialize the ConfigManager
        config_manager = ConfigManager(
            config_path=config_path,
            service_specific_model=AnalysisEngineConfig,
            logger=logger
        )
        
        # Get configuration
        config = config_manager.get_config()
        
        # Print configuration
        logger.info(f"App environment: {config.app.environment}")
        logger.info(f"Is development: {config_manager.is_development()}")
        logger.info(f"Database host: {config_manager.get_database_config().host}")
        logger.info(f"Service port: {config_manager.get_service_config().port}")
        logger.info(f"Market Data Service URL: {config_manager.get_service_client_config('market_data_service').base_url}")
        logger.info(f"Service-specific max workers: {config_manager.get_service_specific_config().max_workers}")
    finally:
        # Clean up
        os.unlink(config_path)


# Run the examples
if __name__ == "__main__":
    example_load_from_file()
    example_override_with_env_vars()
    example_config_manager()
