"""
Standardized Configuration Management Module

This module provides a standardized configuration management system for the platform.
It supports loading configuration from files, environment variables, and defaults,
with validation and type conversion.
"""

import os
import json
import yaml
import logging
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_type_hints

from pydantic import BaseModel, Field, SecretStr, ValidationError, create_model, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict

# Type variable for configuration models
T = TypeVar('T', bound=BaseSettings)

# Logger
logger = logging.getLogger(__name__)


class ConfigSource(str, Enum):
    """Configuration source types."""
    
    FILE = "file"
    ENV = "environment"
    DEFAULT = "default"
    OVERRIDE = "override"


class ConfigValue:
    """
    Configuration value with metadata.
    
    Attributes:
        value: The configuration value
        source: The source of the configuration value
        path: The path to the configuration value
    """
    
    def __init__(
        self,
        value: Any,
        source: ConfigSource,
        path: Optional[str] = None
    ):
        """
        Initialize a configuration value.
        
        Args:
            value: The configuration value
            source: The source of the configuration value
            path: The path to the configuration value
        """
        self.value = value
        self.source = source
        self.path = path
    
    def __repr__(self) -> str:
        """Get string representation."""
        return f"ConfigValue(value={self.value}, source={self.source}, path={self.path})"


class BaseAppSettings(BaseSettings):
    """
    Base application settings shared across services.
    
    This class provides common settings for all services, including:
    - Service metadata (name, version, environment)
    - Logging configuration
    - Database configuration
    - Redis configuration
    - API configuration
    - Security configuration
    """
    
    # Service metadata
    SERVICE_NAME: str = Field("default-service", description="Name of the service")
    SERVICE_VERSION: str = Field("0.1.0", description="Version of the service")
    ENVIRONMENT: str = Field("development", description="Deployment environment")
    
    # Logging configuration
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    LOG_FORMAT: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Logging format"
    )
    LOG_FILE: Optional[str] = Field(None, description="Log file path")
    
    # Debug mode
    DEBUG: bool = Field(False, description="Enable debug mode")
    
    # Database configuration
    DB_DRIVER: str = Field("postgresql+asyncpg", description="Database driver")
    DB_HOST: str = Field("localhost", description="Database host")
    DB_PORT: int = Field(5432, description="Database port")
    DB_NAME: str = Field(..., description="Database name")
    DB_USER: str = Field(..., description="Database user")
    DB_PASSWORD: SecretStr = Field(..., description="Database password")
    DB_POOL_SIZE: int = Field(5, description="Database connection pool size")
    DB_MAX_OVERFLOW: int = Field(10, description="Database connection pool max overflow")
    DB_ECHO: bool = Field(False, description="Echo SQL statements")
    
    # Redis configuration
    REDIS_HOST: str = Field("localhost", description="Redis host")
    REDIS_PORT: int = Field(6379, description="Redis port")
    REDIS_DB: int = Field(0, description="Redis database")
    REDIS_PASSWORD: Optional[SecretStr] = Field(None, description="Redis password")
    
    # API configuration
    API_HOST: str = Field("0.0.0.0", description="API host")
    API_PORT: int = Field(8000, description="API port")
    API_PREFIX: str = Field("/api/v1", description="API prefix")
    API_WORKERS: int = Field(4, description="Number of API workers")
    API_TIMEOUT: int = Field(60, description="API timeout in seconds")
    
    # Security configuration
    API_KEY_NAME: str = Field("X-API-Key", description="API key header name")
    API_KEY: Optional[SecretStr] = Field(None, description="API key")
    JWT_SECRET: Optional[SecretStr] = Field(None, description="JWT secret")
    JWT_ALGORITHM: str = Field("HS256", description="JWT algorithm")
    JWT_EXPIRATION: int = Field(3600, description="JWT expiration in seconds")
    
    # Monitoring configuration
    ENABLE_METRICS: bool = Field(True, description="Enable metrics collection")
    METRICS_PORT: int = Field(9090, description="Metrics port")
    
    # Tracing configuration
    ENABLE_TRACING: bool = Field(False, description="Enable distributed tracing")
    TRACING_EXPORTER: str = Field("jaeger", description="Tracing exporter")
    TRACING_HOST: str = Field("localhost", description="Tracing host")
    TRACING_PORT: int = Field(6831, description="Tracing port")
    
    # Health check configuration
    HEALTH_CHECK_INTERVAL: int = Field(60, description="Health check interval in seconds")
    
    # Model configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    @property
    def database_url(self) -> str:
        """
        Get the database URL.
        
        Returns:
            Database URL
        """
        password = self.DB_PASSWORD.get_secret_value() if self.DB_PASSWORD else ""
        return f"{self.DB_DRIVER}://{self.DB_USER}:{password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    @property
    def redis_url(self) -> str:
        """
        Get the Redis URL.
        
        Returns:
            Redis URL
        """
        password_part = ""
        if self.REDIS_PASSWORD:
            password = self.REDIS_PASSWORD.get_secret_value()
            password_part = f":{password}@"
        return f"redis://{password_part}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    @field_validator("ENVIRONMENT")
    def validate_environment(cls, v: str) -> str:
        """
        Validate the environment.
        
        Args:
            v: Environment value
            
        Returns:
            Validated environment value
            
        Raises:
            ValueError: If the environment is invalid
        """
        valid_environments = ["development", "testing", "staging", "production"]
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of {valid_environments}")
        return v
    
    @field_validator("LOG_LEVEL")
    def validate_log_level(cls, v: str) -> str:
        """
        Validate the log level.
        
        Args:
            v: Log level value
            
        Returns:
            Validated log level value
            
        Raises:
            ValueError: If the log level is invalid
        """
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @field_validator("API_PORT", "DB_PORT", "REDIS_PORT", "METRICS_PORT", "TRACING_PORT")
    def validate_port(cls, v: int, info) -> int:
        """
        Validate a port number.
        
        Args:
            v: Port value
            info: Validation info
            
        Returns:
            Validated port value
            
        Raises:
            ValueError: If the port is invalid
        """
        if v < 1 or v > 65535:
            raise ValueError(f"{info.field_name} must be between 1 and 65535")
        return v
    
    @field_validator("DB_POOL_SIZE", "DB_MAX_OVERFLOW", "API_WORKERS")
    def validate_positive_int(cls, v: int, info) -> int:
        """
        Validate a positive integer.
        
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


class ConfigManager:
    """
    Configuration manager for loading and accessing configuration.
    
    This class provides methods for loading configuration from files,
    environment variables, and defaults, with validation and type conversion.
    """
    
    def __init__(
        self,
        settings_class: Type[T] = BaseAppSettings,
        env_file: Optional[str] = None,
        config_file: Optional[str] = None,
        env_prefix: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the configuration manager.
        
        Args:
            settings_class: Settings class to use
            env_file: Path to environment file
            config_file: Path to configuration file
            env_prefix: Prefix for environment variables
            logger: Logger to use
        """
        self.settings_class = settings_class
        self.env_file = env_file
        self.config_file = config_file
        self.env_prefix = env_prefix
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize settings
        self._settings = None
        self._config_values: Dict[str, ConfigValue] = {}
        
        # Load settings
        self.reload()
    
    def reload(self) -> None:
        """Reload configuration from all sources."""
        # Load configuration from file
        file_config = {}
        if self.config_file:
            try:
                file_config = self._load_config_file(self.config_file)
                for key, value in self._flatten_dict(file_config).items():
                    self._config_values[key] = ConfigValue(
                        value=value,
                        source=ConfigSource.FILE,
                        path=self.config_file
                    )
            except Exception as e:
                self.logger.warning(f"Failed to load configuration file: {e}")
        
        # Create settings with environment variables
        env_settings = {
            "env_file": self.env_file,
            "env_prefix": self.env_prefix
        }
        
        # Create settings
        try:
            self._settings = self.settings_class(**file_config, **env_settings)
            
            # Store environment variable values
            for field_name in self._settings.model_fields:
                if hasattr(self._settings, field_name):
                    value = getattr(self._settings, field_name)
                    
                    # Check if value is from environment
                    env_var = self._get_env_var_name(field_name)
                    if env_var in os.environ:
                        self._config_values[field_name] = ConfigValue(
                            value=value,
                            source=ConfigSource.ENV,
                            path=env_var
                        )
                    elif field_name not in self._config_values:
                        self._config_values[field_name] = ConfigValue(
                            value=value,
                            source=ConfigSource.DEFAULT,
                            path=None
                        )
        except ValidationError as e:
            self.logger.error(f"Configuration validation error: {e}")
            raise
    
    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If the file is not found
            ValueError: If the file format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Load configuration based on file extension
        if file_path.suffix.lower() in (".yaml", ".yml"):
            with open(file_path, "r") as f:
                return yaml.safe_load(f)
        elif file_path.suffix.lower() == ".json":
            with open(file_path, "r") as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
    
    def _get_env_var_name(self, field_name: str) -> str:
        """
        Get the environment variable name for a field.
        
        Args:
            field_name: Field name
            
        Returns:
            Environment variable name
        """
        if self.env_prefix:
            return f"{self.env_prefix}{field_name}"
        return field_name
    
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        separator: str = "."
    ) -> Dict[str, Any]:
        """
        Flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key
            separator: Key separator
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{separator}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, separator).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        if not self._settings:
            self.reload()
        
        if hasattr(self._settings, key):
            return getattr(self._settings, key)
        
        return default
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Dictionary of configuration values
        """
        if not self._settings:
            self.reload()
        
        return self._settings.model_dump()
    
    def get_with_metadata(self, key: str) -> Optional[ConfigValue]:
        """
        Get a configuration value with metadata.
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value with metadata
        """
        if key in self._config_values:
            return self._config_values[key]
        
        if hasattr(self._settings, key):
            return ConfigValue(
                value=getattr(self._settings, key),
                source=ConfigSource.DEFAULT,
                path=None
            )
        
        return None
    
    def get_all_with_metadata(self) -> Dict[str, ConfigValue]:
        """
        Get all configuration values with metadata.
        
        Returns:
            Dictionary of configuration values with metadata
        """
        return self._config_values.copy()
    
    def override(self, key: str, value: Any) -> None:
        """
        Override a configuration value.
        
        Args:
            key: Configuration key
            value: New value
            
        Raises:
            AttributeError: If the key is not found
            ValidationError: If the value is invalid
        """
        if not hasattr(self._settings, key):
            raise AttributeError(f"Configuration key not found: {key}")
        
        # Validate value
        field_info = self._settings.model_fields.get(key)
        if field_info:
            # Create a temporary model with just this field
            temp_model = create_model(
                "TempModel",
                **{key: (field_info.annotation, field_info)}
            )
            
            # Validate value
            try:
                temp_instance = temp_model(**{key: value})
                value = getattr(temp_instance, key)
            except ValidationError as e:
                raise ValidationError(f"Invalid value for {key}: {e}")
        
        # Set value
        setattr(self._settings, key, value)
        
        # Update metadata
        self._config_values[key] = ConfigValue(
            value=value,
            source=ConfigSource.OVERRIDE,
            path=None
        )
    
    @property
    def settings(self) -> T:
        """
        Get the settings instance.
        
        Returns:
            Settings instance
        """
        if not self._settings:
            self.reload()
        
        return self._settings


@lru_cache()
def get_config_manager(
    settings_class: Type[T] = BaseAppSettings,
    env_file: Optional[str] = None,
    config_file: Optional[str] = None,
    env_prefix: Optional[str] = None
) -> ConfigManager:
    """
    Get a cached configuration manager instance.
    
    Args:
        settings_class: Settings class to use
        env_file: Path to environment file
        config_file: Path to configuration file
        env_prefix: Prefix for environment variables
        
    Returns:
        Configuration manager instance
    """
    return ConfigManager(
        settings_class=settings_class,
        env_file=env_file,
        config_file=config_file,
        env_prefix=env_prefix
    )


@lru_cache()
def get_settings(
    settings_class: Type[T] = BaseAppSettings,
    env_file: Optional[str] = None,
    config_file: Optional[str] = None,
    env_prefix: Optional[str] = None
) -> T:
    """
    Get a cached settings instance.
    
    Args:
        settings_class: Settings class to use
        env_file: Path to environment file
        config_file: Path to configuration file
        env_prefix: Prefix for environment variables
        
    Returns:
        Settings instance
    """
    config_manager = get_config_manager(
        settings_class=settings_class,
        env_file=env_file,
        config_file=config_file,
        env_prefix=env_prefix
    )
    return config_manager.settings
