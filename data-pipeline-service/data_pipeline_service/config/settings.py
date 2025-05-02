"""
Configuration settings for the Data Pipeline Service.
"""
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, SecretStr
from common_lib.config import AppSettings, load_settings  # Import base class and loader


# Inherit from the common AppSettings
class DataPipelineSettings(AppSettings):
    """Settings specific to the Data Pipeline Service, inheriting common settings."""

    # --- Service Specific Metadata ---
    # Override SERVICE_NAME from base class
    SERVICE_NAME: str = "data-pipeline-service"
    # Keep app_version if needed, or rely on deployment tags/metadata
    app_version: str = "0.1.0"
    # DEBUG_MODE and LOG_LEVEL are inherited from AppSettings

    # --- API settings ---
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = ["*"]

    # --- Database settings ---
    # DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, DB_POOL_SIZE, DB_SSL_REQUIRED
    # and DATABASE_URL are inherited from AppSettings.
    # The default DATABASE_URL uses asyncpg driver, which is correct for this service.

    # --- Kafka event bus settings ---
    kafka_bootstrap_servers: str = Field(
        "localhost:9092", description="Comma-separated list of Kafka broker addresses"
    )
    kafka_consumer_group_prefix: str = Field(
        "data-pipeline", description="Prefix for Kafka consumer groups"
    )
    kafka_auto_create_topics: bool = Field(
        True, description="Whether to automatically create Kafka topics"
    )
    kafka_producer_acks: str = Field(
        "all", description="Kafka producer acknowledgment setting"
    )

    # --- API Keys for data providers ---
    oanda_api_key: Optional[SecretStr] = Field(
        None, env="OANDA_API_KEY", description="API key for Oanda"
    )
    oanda_account_id: Optional[str] = Field(
        None, env="OANDA_ACCOUNT_ID", description="Account ID for Oanda"
    )

    # --- Data fetching settings ---
    max_requests_per_minute: int = Field(60, description="Max API requests per minute")
    max_retries: int = Field(3, description="Max retries for failed requests")
    retry_delay_seconds: int = Field(5, description="Delay between retries in seconds")
    timeout_seconds: int = Field(30, description="Timeout for API requests in seconds")

    # --- Logging settings ---
    # LOG_LEVEL is inherited. Validation is also handled in the base class.

    # --- Redis settings ---
    # REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD, REDIS_TIMEOUT, REDIS_URL
    # are inherited from AppSettings.

    # --- Optional object storage settings ---
    use_object_storage: bool = Field(
        False, env="USE_OBJECT_STORAGE", description="Whether to use object storage"
    )
    object_storage_endpoint: Optional[str] = Field(
        None, env="OBJECT_STORAGE_ENDPOINT", description="S3 endpoint"
    )
    object_storage_key: Optional[SecretStr] = Field(
        None, env="OBJECT_STORAGE_KEY", description="S3 access key"
    )
    object_storage_secret: Optional[SecretStr] = Field(
        None, env="OBJECT_STORAGE_SECRET", description="S3 secret key"
    )
    object_storage_bucket: Optional[str] = Field(
        None, env="OBJECT_STORAGE_BUCKET", description="S3 bucket name"
    )

    # --- Pydantic Settings Configuration ---
    # model_config is inherited and configured in AppSettings
    # It already includes env_file loading and case_insensitive settings.


# Remove original Settings class definition and validation if fully covered by base class
# class Settings(BaseSettings): ...
# @field_validator("log_level") ... - Handled in base class

# Update the getter function to use the new class and the common loader
@lru_cache()  # Keep lru_cache here or rely on the one in load_settings
def get_settings() -> DataPipelineSettings:
    """Get cached settings instance for Data Pipeline Service."""
    # Use the common load_settings function with the specific settings class
    return load_settings(DataPipelineSettings)

# Optional: Instantiate settings immediately if needed globally (though getter is preferred)
# settings = get_settings()