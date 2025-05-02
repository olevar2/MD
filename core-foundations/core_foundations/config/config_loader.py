"""
Configuration Loader Module.

Provides configuration loading from YAML/JSON files with schema validation via Pydantic.
Features include environment variable support, config merging, and comprehensive validation.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
import re

import yaml
from pydantic import BaseModel, ValidationError

from core_foundations.utils.logger import get_logger
from common_lib.exceptions import ConfigurationError, ConfigNotFoundError, ConfigValidationError

logger = get_logger("config_loader")

# Generic type for config model
T = TypeVar('T', bound=BaseModel)


class ConfigLoader(Generic[T]):
    """
    Configuration loader with validation.
    
    Loads configuration from YAML/JSON files and validates using Pydantic models.
    Features include environment variable support, config merging, and comprehensive validation.
    """
    def __init__(self, config_model: Type[T], env_prefix: str = "APP_"):
        """
        Initialize the config loader.
        
        Args:
            config_model: Pydantic model class for configuration validation
            env_prefix: Prefix for environment variables (set to empty string to use exact case match)
        """
        self.config_model = config_model
        self.env_prefix = env_prefix
    
    def load_from_file(self, file_path: str) -> T:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file (YAML or JSON)
            
        Returns:
            Validated configuration object
            
        Raises:
            ConfigNotFoundError: If the file doesn't exist
            ConfigurationError: If there's an error loading the file
            ConfigValidationError: If the configuration is invalid
        """
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            logger.error(f"Config file not found: {file_path}")
            raise ConfigNotFoundError(file_path)
        
        # Load based on file extension
        file_extension = path.suffix.lower()
        
        try:
            if file_extension == '.yaml' or file_extension == '.yml':
                with open(path, 'r') as file:
                    config_data = yaml.safe_load(file)
            elif file_extension == '.json':
                with open(path, 'r') as file:
                    config_data = json.load(file)
            else:
                logger.error(f"Unsupported file extension: {file_extension}")
                raise ConfigurationError(
                    f"Unsupported file extension: {file_extension}. Use .yaml, .yml, or .json",
                    error_code="UNSUPPORTED_CONFIG_FORMAT"
                )
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {str(e)}")
            raise ConfigurationError(
                f"Failed to parse YAML config: {str(e)}",
                error_code="YAML_PARSE_ERROR"
            ) from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON config: {str(e)}")
            raise ConfigurationError(
                f"Failed to parse JSON config: {str(e)}",
                error_code="JSON_PARSE_ERROR"
            ) from e
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
            raise ConfigurationError(
                f"Error loading config file: {str(e)}",
                error_code="CONFIG_LOAD_ERROR"
            ) from e
        
        # Validate configuration
        try:
            return self.load_from_dict(config_data)
        except ValidationError as e:
            logger.error(f"Config validation error: {str(e)}")
            raise ConfigValidationError(errors=e.errors()) from e
    
    def load_from_dict(self, config_data: Dict[str, Any]) -> T:
        """
        Load configuration from a dictionary.
        
        Args:
            config_data: Dictionary containing configuration data
            
        Returns:
            Validated configuration object
            
        Raises:
            ConfigValidationError: If the configuration is invalid
        """
        try:
            return self.config_model(**config_data)
        except ValidationError as e:
            logger.error(f"Config validation error: {str(e)}")
            raise ConfigValidationError(errors=e.errors()) from e
    
    def load_from_env(self) -> T:
        """
        Load configuration from environment variables.
        
        Environment variables are converted to config keys by:
        1. Removing the prefix (if any)
        2. Converting to lowercase
        3. Converting underscores to nested keys
        
        Example:
            APP_DATABASE_HOST -> database.host
        
        Returns:
            Validated configuration object
        
        Raises:
            ConfigValidationError: If the configuration is invalid
        """
        config_data = {}
        prefix_len = len(self.env_prefix)
        
        # Extract field info from model
        model_fields = self.config_model.model_fields
        
        for env_key, env_value in os.environ.items():
            # Check if the variable matches our prefix
            if self.env_prefix and not env_key.startswith(self.env_prefix):
                continue
            
            # Remove prefix and convert to lowercase if not using exact match
            if self.env_prefix:
                key = env_key[prefix_len:].lower()
            else:
                key = env_key
            
            # Handle nested keys
            if "_" in key:
                parts = key.split("_")
                current = config_data
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    if not isinstance(current[part], dict):
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = self._convert_env_value(env_value, parts[-1], model_fields)
            else:
                config_data[key] = self._convert_env_value(env_value, key, model_fields)
        
        # Validate configuration
        try:
            return self.load_from_dict(config_data)
        except ValidationError as e:
            logger.error(f"Config validation error from environment variables: {str(e)}")
            raise ConfigValidationError(errors=e.errors()) from e
    
    def _convert_env_value(self, value: str, field_name: str, model_fields: Dict[str, Any]) -> Any:
        """
        Convert environment variable string value to appropriate type.
        
        Args:
            value: String value from environment variable
            field_name: Name of the field
            model_fields: Dictionary of model fields
            
        Returns:
            Converted value with appropriate type
        """
        # If we have field type information, use it for conversion
        if field_name in model_fields:
            field = model_fields[field_name]
            field_type = getattr(field, "annotation", None)
            
            if field_type is bool or (hasattr(field_type, "__origin__") and field_type.__origin__ is Union and bool in field_type.__args__):
                return value.lower() in ("true", "yes", "1", "t", "y")
            
            elif field_type is int or (hasattr(field_type, "__origin__") and field_type.__origin__ is Union and int in field_type.__args__):
                return int(value)
                
            elif field_type is float or (hasattr(field_type, "__origin__") and field_type.__origin__ is Union and float in field_type.__args__):
                return float(value)
                
            elif field_type is list or field_type is List or (hasattr(field_type, "__origin__") and field_type.__origin__ is list):
                return [item.strip() for item in value.split(",")]
        
        # Try to infer type from value
        if value.lower() in ("true", "false", "yes", "no", "t", "f", "y", "n"):
            return value.lower() in ("true", "yes", "t", "y")
            
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            # Return as string if we can't convert
            return value
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.
        
        Values in override_config take precedence over values in base_config.
        Nested dictionaries are merged recursively.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override values in base
            
        Returns:
            Merged configuration dictionary
        """
        result = base_config.copy()
        
        for key, value in override_config.items():
            # If both are dictionaries, merge them recursively
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def load_with_overrides(self, base_path: str, override_paths: List[str] = None, env: bool = True) -> T:
        """
        Load configuration with support for overrides.
        
        Args:
            base_path: Path to base configuration file
            override_paths: List of override configuration file paths
            env: Whether to include environment variables as final override
            
        Returns:
            Validated merged configuration
            
        Raises:
            ConfigNotFoundError: If the base file doesn't exist
            ConfigurationError: If there's an error loading a file
            ConfigValidationError: If the final configuration is invalid
        """
        # Load base configuration
        config_data = {}
        if base_path:
            base_config = self.load_from_file(base_path)
            config_data = base_config.model_dump()
        
        # Apply each override
        if override_paths:
            for path in override_paths:
                if path and Path(path).exists():
                    override_config = self.load_from_file(path)
                    config_data = self.merge_configs(config_data, override_config.model_dump())
        
        # Apply environment variables as final override if requested
        if env:
            try:
                env_config = self.load_from_env()
                config_data = self.merge_configs(config_data, env_config.model_dump())
            except ValidationError:
                # We'll validate the final config at the end, so just log here
                logger.warning("Invalid environment configuration, skipping environment override")
        
        # Validate final configuration
        try:
            return self.load_from_dict(config_data)
        except ValidationError as e:
            logger.error(f"Config validation error in merged configuration: {str(e)}")
            raise ConfigValidationError(errors=e.errors()) from e


def create_config_loader(config_model: Type[T]) -> ConfigLoader[T]:
    """
    Helper function to create a config loader for a specific model.
    
    Args:
        config_model: Pydantic model class for configuration validation
        
    Returns:
        ConfigLoader instance for the specified model
    """
    return ConfigLoader(config_model)