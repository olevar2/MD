"""
Configuration settings for the ML Integration Service.
"""
from typing import Optional
from functools import lru_cache

from pydantic import Field, SecretStr  # Keep Field, add SecretStr if needed
from common_lib.config import AppSettings, load_settings  # Import base class and loader


# Inherit from the common AppSettings
class MLIntegrationSettings(AppSettings):
    """Settings specific to the ML Integration Service."""

    # --- Service Specific Metadata ---
    # Override SERVICE_NAME from base class
    SERVICE_NAME: str = "ml-integration-service"
    # DEBUG_MODE and LOG_LEVEL are inherited

    # --- API settings ---
    API_PREFIX: str = "/api/v1"

    # --- Server settings ---
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8080, env="PORT")

    # --- ML Workbench service integration ---
    ML_WORKBENCH_API_URL: str = Field(
        default="http://ml-workbench-service:8000/api/v1",
        env="ML_WORKBENCH_API_URL",
    )
    ML_WORKBENCH_API_KEY: Optional[SecretStr] = Field(
        default=None,
        env="ML_WORKBENCH_API_KEY",
    )

    # --- Analysis Engine service integration ---
    ANALYSIS_ENGINE_API_URL: str = Field(
        default="http://analysis-engine-service:8000/api/v1",
        env="ANALYSIS_ENGINE_API_URL",
    )
    ANALYSIS_ENGINE_API_KEY: Optional[SecretStr] = Field(
        default=None,
        env="ANALYSIS_ENGINE_API_KEY",
    )

    # --- Strategy execution engine integration ---
    STRATEGY_EXECUTION_API_URL: str = Field(
        default="http://strategy-execution-engine:8000/api/v1",
        env="STRATEGY_EXECUTION_API_URL",
    )
    STRATEGY_EXECUTION_API_KEY: Optional[SecretStr] = Field(
        default=None,
        env="STRATEGY_EXECUTION_API_KEY",
    )

    # --- Model registry settings ---
    MLFLOW_TRACKING_URI: str = Field(
        default="http://mlflow:5000",
        env="MLFLOW_TRACKING_URI",
    )

    # --- Database settings ---
    # DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, DB_POOL_SIZE, DB_SSL_REQUIRED
    # and the computed DATABASE_URL are inherited from AppSettings.
    # Ensure environment variables like DB_USER, DB_PASSWORD etc. are set for this service.
    # Remove the old DATABASE_URL field:
    # DATABASE_URL: str = Field(...)

    # --- API Authentication ---
    # API_KEY_NAME and API_KEY (as Optional[SecretStr]) are inherited.
    # Ensure API_KEY env var is set if needed.
    # Remove the old API_KEY field:
    # API_KEY: str = Field(...)

    # --- Cache settings ---
    # REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD, REDIS_TIMEOUT
    # and the computed REDIS_URL are inherited from AppSettings.
    # Ensure environment variables like REDIS_HOST, REDIS_PASSWORD etc. are set.
    # Remove the old REDIS_URL field:
    # REDIS_URL: str = Field(...)
    CACHE_TTL_SECONDS: int = Field(default=3600, env="CACHE_TTL_SECONDS")

    # --- Pydantic Settings Configuration ---
    # model_config is inherited from AppSettings.
    # Remove the old Config class:
    # class Config:
    #     env_file = ".env"
    #     case_sensitive = True # Base class uses case_sensitive=False


# Remove original ServiceSettings class definition
# class ServiceSettings(BaseSettings): ...

# Update the getter function (or instantiation) to use the new class and the common loader
@lru_cache()
def get_settings() -> MLIntegrationSettings:
    """Get cached settings instance for ML Integration Service."""
    return load_settings(MLIntegrationSettings)


# Instantiate settings using the getter
settings = get_settings()
