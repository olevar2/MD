"""
Base configuration settings using Pydantic BaseSettings.

Loads configuration from environment variables and .env files.
Provides common settings fields used across services.
"""
import os
from functools import lru_cache
from typing import Optional, Type, TypeVar

from pydantic import Field, SecretStr, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

T = TypeVar('T', bound='AppSettings') # Type variable for settings classes

class AppSettings(BaseSettings):
    """Base application settings shared across services."""

    # --- Service Metadata ---
    SERVICE_NAME: str = Field("default-service", description="Name of the service")
    DEBUG_MODE: bool = Field(default=False, env="DEBUG_MODE", description="Enable debug mode")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL", description="Logging level")

    # --- Database Settings ---
    # Environment variables are expected (e.g., DB_USER=...)
    DB_USER: str = Field(..., env="DB_USER", description="Database user")
    DB_PASSWORD: SecretStr = Field(..., env="DB_PASSWORD", description="Database password")
    DB_HOST: str = Field(..., env="DB_HOST", description="Database host")
    DB_PORT: int = Field(5432, env="DB_PORT", description="Database port")
    DB_NAME: str = Field(..., env="DB_NAME", description="Database name")
    DB_POOL_SIZE: int = Field(10, env="DB_POOL_SIZE", description="Database connection pool size")
    DB_SSL_REQUIRED: bool = Field(False, env="DB_SSL_REQUIRED", description="Require SSL for DB connection")

    @computed_field(description="SQLAlchemy compatible database connection URL (asyncpg driver)")
    @property
    def DATABASE_URL(self) -> str:
        """Construct the database connection URL using asyncpg driver."""
        password = self.DB_PASSWORD.get_secret_value()
        # Default to asyncpg driver, services can override if needed by redefining
        # or by using the individual DB_* fields with a different driver.
        return f"postgresql+asyncpg://{self.DB_USER}:{password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # --- Redis Settings ---
    REDIS_HOST: str = Field("localhost", env="REDIS_HOST", description="Redis host")
    REDIS_PORT: int = Field(6379, env="REDIS_PORT", description="Redis port")
    REDIS_DB: int = Field(0, env="REDIS_DB", description="Redis database index")
    REDIS_PASSWORD: Optional[SecretStr] = Field(None, env="REDIS_PASSWORD", description="Redis password")
    REDIS_TIMEOUT: int = Field(5, env="REDIS_TIMEOUT", description="Redis connection timeout")

    @computed_field(description="Redis connection URL")
    @property
    def REDIS_URL(self) -> str:
        """Construct the Redis connection URL."""
        password_part = ""
        if self.REDIS_PASSWORD:
            password = self.REDIS_PASSWORD.get_secret_value()
            password_part = f":{password}@"
        return f"redis://{password_part}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # --- API Authentication ---
    API_KEY_NAME: str = Field(default="X-API-Key", env="API_KEY_NAME", description="HTTP header name for API Key")
    API_KEY: Optional[SecretStr] = Field(default=None, env="API_KEY", description="API Key for securing internal endpoints")

    # --- Pydantic Settings Configuration ---
    model_config = SettingsConfigDict(
        env_file=".env",             # Load .env file if present
        env_file_encoding="utf-8",
        case_sensitive=False,      # Environment variables are typically uppercase
        extra='ignore'             # Allow services to define extra fields without error
    )

    # --- Validation ---
    @field_validator("LOG_LEVEL")
    def validate_log_level(cls, v: str) -> str:
        """Validate the log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        level = v.upper()
        if level not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}, got '{v}'")
        return level


# --- Utility Function ---

# Use lru_cache for simplicity and thread-safety provided by functools
@lru_cache()
def load_settings(settings_cls: Type[T]) -> T:
    """
    Loads and caches the settings instance for the given settings class.

    Args:
        settings_cls: The Pydantic BaseSettings class (subclass of AppSettings) to instantiate.

    Returns:
        A cached instance of the provided settings class.
    """
    # The lru_cache decorator handles the caching based on the settings_cls argument
    try:
        return settings_cls()
    except Exception as e:
        # Provide more context if settings fail to load (e.g., missing env vars)
        print(f"Error loading settings for {settings_cls.__name__}: {e}")
        raise

