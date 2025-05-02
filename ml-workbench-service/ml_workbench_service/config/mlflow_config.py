"""
MLflow Configuration for Experiment Tracking

This module contains configurations for MLflow experiment tracking,
including tracking URI, artifact storage, and default experiment settings.
"""
from pydantic import BaseSettings
from typing import Optional, Dict, Any
import os


class MLflowSettings(BaseSettings):
    """
    Configuration settings for MLflow integration.
    
    Attributes:
        mlflow_tracking_uri: URI for MLflow tracking server
        mlflow_artifact_location: Base location for artifact storage
        default_experiment_name: Default experiment name if none is provided
        experiment_tags: Default tags to apply to all experiments
        registry_uri: URI for MLflow model registry
        s3_endpoint_url: Optional S3 endpoint URL for artifact storage
        s3_ignore_tls: Whether to ignore TLS for S3 connections
    """
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"  # Default to local SQLite DB
    mlflow_artifact_location: str = "file:./mlruns"  # Default to local filesystem
    default_experiment_name: str = "forex_trading_platform_experiment"
    experiment_tags: Dict[str, str] = {"project": "forex_trading_platform"}
    registry_uri: Optional[str] = None
    s3_endpoint_url: Optional[str] = None
    s3_ignore_tls: bool = False
    
    # Environment-specific configurations
    def get_environment_config(self) -> Dict[str, Any]:
        """
        Get environment-specific configurations based on ENVIRONMENT variable.
        
        Returns:
            Dict[str, Any]: Environment-specific configurations
        """
        env = os.getenv("ENVIRONMENT", "development")
        
        if env == "production":
            return {
                "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", self.mlflow_tracking_uri),
                "mlflow_artifact_location": os.getenv("MLFLOW_ARTIFACT_LOCATION", self.mlflow_artifact_location),
                "registry_uri": os.getenv("MLFLOW_REGISTRY_URI", self.registry_uri),
                "s3_endpoint_url": os.getenv("MLFLOW_S3_ENDPOINT_URL", self.s3_endpoint_url),
                "s3_ignore_tls": os.getenv("MLFLOW_S3_IGNORE_TLS", "").lower() == "true"
            }
        
        if env == "staging":
            return {
                "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow_staging.db"),
                "mlflow_artifact_location": os.getenv("MLFLOW_ARTIFACT_LOCATION", "file:./mlruns_staging"),
                "registry_uri": os.getenv("MLFLOW_REGISTRY_URI", self.registry_uri),
                "s3_endpoint_url": os.getenv("MLFLOW_S3_ENDPOINT_URL", self.s3_endpoint_url),
                "s3_ignore_tls": os.getenv("MLFLOW_S3_IGNORE_TLS", "").lower() == "true"
            }
        
        # Default to development settings
        return {
            "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", self.mlflow_tracking_uri),
            "mlflow_artifact_location": os.getenv("MLFLOW_ARTIFACT_LOCATION", self.mlflow_artifact_location),
            "registry_uri": os.getenv("MLFLOW_REGISTRY_URI", self.registry_uri),
            "s3_endpoint_url": os.getenv("MLFLOW_S3_ENDPOINT_URL", self.s3_endpoint_url),
            "s3_ignore_tls": os.getenv("MLFLOW_S3_IGNORE_TLS", "").lower() == "true"
        }

    def update_from_env(self) -> 'MLflowSettings':
        """
        Update settings from environment variables.
        
        Returns:
            MLflowSettings: Updated settings instance
        """
        env_config = self.get_environment_config()
        for key, value in env_config.items():
            setattr(self, key, value)
        return self


# Create a default settings instance
mlflow_settings = MLflowSettings().update_from_env()