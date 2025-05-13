"""
Configuration Manager Module

This module provides a singleton configuration manager for the application.
"""

import os
import logging
from typing import Dict, Any, Optional, Type, Union, ClassVar

from common_lib.config.config_schema import Config, ServiceSpecificConfig
from common_lib.config.config_loader import ConfigLoader


class ConfigManager:
    """
    Configuration manager.
    
    This class provides a singleton configuration manager for the application.
    """
    
    _instance: ClassVar[Optional["ConfigManager"]] = None
    _config: Optional[Config] = None
    
    def __new__(cls, *args, **kwargs):
        """
        Create a new instance of the configuration manager.
        
        Returns:
            Singleton instance of the configuration manager
        """
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        service_specific_model: Optional[Type[ServiceSpecificConfig]] = None,
        env_prefix: str = "APP_",
        default_config_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
            service_specific_model: Model for service-specific configuration
            env_prefix: Prefix for environment variables
            default_config_path: Path to the default configuration file
            logger: Logger to use (if None, creates a new logger)
        """
        # Skip initialization if already initialized
        if getattr(self, "_initialized", False):
            return
        
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config_loader = ConfigLoader(logger=self.logger)
        
        # Load configuration if config_path is provided
        if config_path is not None:
            self.load_config(
                config_path,
                service_specific_model,
                env_prefix,
                default_config_path
            )
        
        self._initialized = True
    
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
        self._config = self.config_loader.load_config(
            config_path,
            service_specific_model,
            env_prefix,
            default_config_path
        )
        return self._config
    
    def get_config(self) -> Config:
        """
        Get the loaded configuration.
        
        Returns:
            Loaded configuration
            
        Raises:
            RuntimeError: If configuration is not loaded
        """
        if self._config is None:
            self.logger.error("Configuration not loaded")
            raise RuntimeError("Configuration not loaded")
        return self._config
    
    def get_service_specific_config(self) -> ServiceSpecificConfig:
        """
        Get the service-specific configuration.
        
        Returns:
            Service-specific configuration
            
        Raises:
            RuntimeError: If configuration is not loaded or service-specific configuration is not set
        """
        config = self.get_config()
        if config.service_specific is None:
            self.logger.error("Service-specific configuration not set")
            raise RuntimeError("Service-specific configuration not set")
        return config.service_specific
    
    def get_app_config(self):
        """
        Get the application configuration.
        
        Returns:
            Application configuration
            
        Raises:
            RuntimeError: If configuration is not loaded
        """
        return self.get_config().app
    
    def get_database_config(self):
        """
        Get the database configuration.
        
        Returns:
            Database configuration
            
        Raises:
            RuntimeError: If configuration is not loaded
        """
        return self.get_config().database
    
    def get_logging_config(self):
        """
        Get the logging configuration.
        
        Returns:
            Logging configuration
            
        Raises:
            RuntimeError: If configuration is not loaded
        """
        return self.get_config().logging
    
    def get_service_config(self):
        """
        Get the service configuration.
        
        Returns:
            Service configuration
            
        Raises:
            RuntimeError: If configuration is not loaded
        """
        return self.get_config().service
    
    def get_service_clients_config(self):
        """
        Get the service clients configuration.
        
        Returns:
            Service clients configuration
            
        Raises:
            RuntimeError: If configuration is not loaded
        """
        return self.get_config().service_clients
    
    def get_service_client_config(self, service_name: str):
        """
        Get the configuration for a specific service client.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Service client configuration
            
        Raises:
            RuntimeError: If configuration is not loaded
            KeyError: If service client configuration is not found
        """
        service_clients = self.get_service_clients_config()
        if not hasattr(service_clients, service_name):
            self.logger.error(f"Service client configuration not found: {service_name}")
            raise KeyError(f"Service client configuration not found: {service_name}")
        return getattr(service_clients, service_name)
    
    def is_development(self) -> bool:
        """
        Check if the application is running in development mode.
        
        Returns:
            True if the application is running in development mode, False otherwise
            
        Raises:
            RuntimeError: If configuration is not loaded
        """
        return self.get_app_config().environment == "development"
    
    def is_testing(self) -> bool:
        """
        Check if the application is running in testing mode.
        
        Returns:
            True if the application is running in testing mode, False otherwise
            
        Raises:
            RuntimeError: If configuration is not loaded
        """
        return self.get_app_config().environment == "testing" or self.get_app_config().testing
    
    def is_staging(self) -> bool:
        """
        Check if the application is running in staging mode.
        
        Returns:
            True if the application is running in staging mode, False otherwise
            
        Raises:
            RuntimeError: If configuration is not loaded
        """
        return self.get_app_config().environment == "staging"
    
    def is_production(self) -> bool:
        """
        Check if the application is running in production mode.
        
        Returns:
            True if the application is running in production mode, False otherwise
            
        Raises:
            RuntimeError: If configuration is not loaded
        """
        return self.get_app_config().environment == "production"