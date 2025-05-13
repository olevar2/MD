"""
Standardized Configuration Module for ML Integration Service

This module provides configuration management for the service using the standardized
configuration management system from common-lib.
"""

import os
from functools import lru_cache
from typing import Optional, List, Dict, Any

from pydantic import Field, field_validator, SecretStr

from common_lib.config import BaseAppSettings, get_settings, get_config_manager


class MLIntegrationServiceSettings(BaseAppSettings):
    """
    ML Integration Service-specific settings.
    
    This class extends the base application settings with service-specific configuration.
    """
    
    # Override service name
    SERVICE_NAME: str = Field("ml-integration-service", description="Name of the service")
    
    # API configuration
    API_PREFIX: str = Field("/api/v1", description="API prefix")
    
    # Server settings
    HOST: str = Field("0.0.0.0", description="Host to bind to")
    PORT: int = Field(8080, description="Port to bind to")
    
    # ML Workbench service integration
    ML_WORKBENCH_API_URL: str = Field(
        "http://ml-workbench-service:8000/api/v1",
        description="URL of the ML Workbench service"
    )
    ML_WORKBENCH_API_KEY: Optional[SecretStr] = Field(
        None,
        description="API key for the ML Workbench service"
    )
    
    # Analysis Engine service integration
    ANALYSIS_ENGINE_API_URL: str = Field(
        "http://analysis-engine-service:8000/api/v1",
        description="URL of the Analysis Engine service"
    )
    ANALYSIS_ENGINE_API_KEY: Optional[SecretStr] = Field(
        None,
        description="API key for the Analysis Engine service"
    )
    
    # Strategy execution engine integration
    STRATEGY_EXECUTION_API_URL: str = Field(
        "http://strategy-execution-engine:8000/api/v1",
        description="URL of the Strategy Execution Engine"
    )
    STRATEGY_EXECUTION_API_KEY: Optional[SecretStr] = Field(
        None,
        description="API key for the Strategy Execution Engine"
    )
    
    # Model registry settings
    MLFLOW_TRACKING_URI: str = Field(
        "http://mlflow:5000",
        description="MLflow tracking URI"
    )
    
    # Cache settings
    CACHE_TTL_SECONDS: int = Field(
        3600,
        description="Cache time-to-live in seconds"
    )
    
    # Model settings
    MODEL_CACHE_ENABLED: bool = Field(
        True,
        description="Whether to enable model caching"
    )
    MODEL_CACHE_TTL_SECONDS: int = Field(
        3600,
        description="Model cache time-to-live in seconds"
    )
    MODEL_REGISTRY_PATH: str = Field(
        "/app/models",
        description="Path to the model registry"
    )
    
    # Feature extraction settings
    FEATURE_EXTRACTION_BATCH_SIZE: int = Field(
        1000,
        description="Batch size for feature extraction"
    )
    FEATURE_EXTRACTION_MAX_WORKERS: int = Field(
        4,
        description="Maximum number of workers for feature extraction"
    )
    
    # Model optimization settings
    OPTIMIZATION_MAX_TRIALS: int = Field(
        100,
        description="Maximum number of trials for model optimization"
    )
    OPTIMIZATION_TIMEOUT: int = Field(
        3600,
        description="Timeout for model optimization in seconds"
    )
    
    # Validation settings
    VALIDATION_SPLIT: float = Field(
        0.2,
        description="Validation split ratio"
    )
    VALIDATION_METRICS: List[str] = Field(
        ["accuracy", "precision", "recall", "f1"],
        description="Validation metrics"
    )
    
    # Reconciliation settings
    RECONCILIATION_ENABLED: bool = Field(
        True,
        description="Whether to enable model reconciliation"
    )
    RECONCILIATION_INTERVAL: int = Field(
        3600,
        description="Interval for model reconciliation in seconds"
    )
    RECONCILIATION_THRESHOLD: float = Field(
        0.05,
        description="Threshold for model reconciliation"
    )
    
    # Add validation
    @field_validator("PORT")
    def validate_port(cls, v: int) -> int:
        """
        Validate port number.
        
        Args:
            v: Port number
            
        Returns:
            Validated port number
            
        Raises:
            ValueError: If the port number is invalid
        """
        if v < 1 or v > 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @field_validator("FEATURE_EXTRACTION_BATCH_SIZE", "FEATURE_EXTRACTION_MAX_WORKERS", "OPTIMIZATION_MAX_TRIALS", "OPTIMIZATION_TIMEOUT")
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
    
    @field_validator("VALIDATION_SPLIT")
    def validate_validation_split(cls, v: float) -> float:
        """
        Validate validation split ratio.
        
        Args:
            v: Validation split ratio
            
        Returns:
            Validated validation split ratio
            
        Raises:
            ValueError: If the validation split ratio is invalid
        """
        if v <= 0 or v >= 1:
            raise ValueError("Validation split must be between 0 and 1")
        return v
    
    @field_validator("RECONCILIATION_THRESHOLD")
    def validate_reconciliation_threshold(cls, v: float) -> float:
        """
        Validate reconciliation threshold.
        
        Args:
            v: Reconciliation threshold
            
        Returns:
            Validated reconciliation threshold
            
        Raises:
            ValueError: If the reconciliation threshold is invalid
        """
        if v <= 0 or v >= 1:
            raise ValueError("Reconciliation threshold must be between 0 and 1")
        return v
    
    @field_validator("VALIDATION_METRICS")
    def validate_validation_metrics(cls, v: List[str]) -> List[str]:
        """
        Validate validation metrics.
        
        Args:
            v: Validation metrics
            
        Returns:
            Validated validation metrics
            
        Raises:
            ValueError: If the validation metrics are invalid
        """
        valid_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "mse", "mae", "rmse", "r2"]
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid validation metric: {metric}. Valid metrics: {valid_metrics}")
        return v


@lru_cache()
def get_service_settings() -> MLIntegrationServiceSettings:
    """
    Get cached service settings.
    
    Returns:
        Service settings
    """
    return get_settings(
        settings_class=MLIntegrationServiceSettings,
        env_file=os.environ.get("ENV_FILE", ".env"),
        config_file=os.environ.get("CONFIG_FILE", "config/config.yaml"),
        env_prefix=os.environ.get("ENV_PREFIX", "ML_INTEGRATION_")
    )


@lru_cache()
def get_service_config_manager():
    """
    Get cached service configuration manager.
    
    Returns:
        Service configuration manager
    """
    return get_config_manager(
        settings_class=MLIntegrationServiceSettings,
        env_file=os.environ.get("ENV_FILE", ".env"),
        config_file=os.environ.get("CONFIG_FILE", "config/config.yaml"),
        env_prefix=os.environ.get("ENV_PREFIX", "ML_INTEGRATION_")
    )


# Create a settings instance for easy access
settings = get_service_settings()
