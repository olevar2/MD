"""
Configuration Loader Module

This module provides utilities for loading and validating configuration.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Type, Union
from pathlib import Path

from pydantic import BaseModel, ValidationError

from common_lib.config.config_schema import Config, ServiceSpecificConfig


class ConfigLoader:
    """
    Configuration loader.
    
    This class provides utilities for loading and validating configuration from
    various sources.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the configuration loader.
        
        Args:
            logger: Logger to use (if None, creates a new logger)
        """
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def load_config(
        self,
        config_path: str,
        service_specific_model: Optional[Type[ServiceSpecificConfig]] = None,
        env_prefix: str = "APP_",
        default_config_path: Optional[str] = None
    ) -> Config:
        """
        Load configuration from a file and environment variables.
        
        Args:
            config_path: Path to the configuration file
            service_specific_model: Model for service-specific configuration
            env_prefix: Prefix for environment variables
            default_config_path: Path to the default configuration file
            
        Returns:
            Loaded and validated configuration
            
        Raises:
            FileNotFoundError: If the configuration file is not found
            ValidationError: If the configuration is invalid
        """
        # Load configuration from file
        config_dict = self._load_config_from_file(config_path, default_config_path)
        
        # Override with environment variables
        config_dict = self._override_with_env_vars(config_dict, env_prefix)
        
        # Validate configuration
        try:
            config = Config.parse_obj(config_dict)
        except ValidationError as e:
            self.logger.error(f"Invalid configuration: {str(e)}")
            raise
        
        # Load service-specific configuration
        if service_specific_model is not None and "service_specific" in config_dict:
            try:
                config.service_specific = service_specific_model.parse_obj(
                    config_dict["service_specific"]
                )
            except ValidationError as e:
                self.logger.error(f"Invalid service-specific configuration: {str(e)}")
                raise
        
        return config
    
    def _load_config_from_file(
        self,
        config_path: str,
        default_config_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file
            default_config_path: Path to the default configuration file
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            FileNotFoundError: If the configuration file is not found
        """
        # Check if the configuration file exists
        if not os.path.exists(config_path):
            if default_config_path is not None and os.path.exists(default_config_path):
                self.logger.warning(
                    f"Configuration file {config_path} not found, using default configuration"
                )
                config_path = default_config_path
            else:
                self.logger.error(f"Configuration file {config_path} not found")
                raise FileNotFoundError(f"Configuration file {config_path} not found")
        
        # Load configuration from file
        file_extension = os.path.splitext(config_path)[1].lower()
        
        try:
            if file_extension in [".yaml", ".yml"]:
                with open(config_path, "r") as f:
                    config_dict = yaml.safe_load(f)
            elif file_extension == ".json":
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
            else:
                self.logger.error(f"Unsupported configuration file format: {file_extension}")
                raise ValueError(f"Unsupported configuration file format: {file_extension}")
        except Exception as e:
            self.logger.error(f"Error loading configuration file: {str(e)}")
            raise
        
        return config_dict
    
    def _override_with_env_vars(
        self,
        config_dict: Dict[str, Any],
        env_prefix: str
    ) -> Dict[str, Any]:
        """
        Override configuration with environment variables.
        
        Args:
            config_dict: Configuration dictionary
            env_prefix: Prefix for environment variables
            
        Returns:
            Updated configuration dictionary
        """
        # Create a copy of the configuration dictionary
        config_dict = config_dict.copy()
        
        # Override with environment variables
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(env_prefix):].lower()
                
                # Split by double underscore to get nested keys
                keys = config_key.split("__")
                
                # Update configuration dictionary
                self._update_nested_dict(config_dict, keys, value)
        
        return config_dict
    
    def _update_nested_dict(
        self,
        d: Dict[str, Any],
        keys: list,
        value: Any
    ) -> None:
        """
        Update a nested dictionary with a value.
        
        Args:
            d: Dictionary to update
            keys: List of keys to traverse
            value: Value to set
        """
        if len(keys) == 1:
            # Convert value to appropriate type
            d[keys[0]] = self._convert_value(value)
        else:
            # Create nested dictionary if it doesn't exist
            if keys[0] not in d:
                d[keys[0]] = {}
            
            # Update nested dictionary
            self._update_nested_dict(d[keys[0]], keys[1:], value)
    
    def _convert_value(self, value: str) -> Any:
        """
        Convert a string value to an appropriate type.
        
        Args:
            value: String value to convert
            
        Returns:
            Converted value
        """
        # Try to convert to boolean
        if value.lower() in ["true", "yes", "1"]:
            return True
        elif value.lower() in ["false", "no", "0"]:
            return False
        
        # Try to convert to integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value