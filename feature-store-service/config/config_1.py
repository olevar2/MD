"""
Configuration module for the Feature Store Caching system.

This module provides configuration options and loading functionality
for the multi-tiered caching system.
"""
import os
import json
from typing import Dict, Any


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class CacheConfig:
    """Configuration handler for the caching system."""
    DEFAULT_CONFIG = {'memory_cache_size': 1000000000, 'memory_cache_ttl': 
        300, 'use_disk_cache': True, 'disk_cache_path': 'cache',
        'disk_cache_size': 50000000000, 'disk_cache_ttl': 86400,
        'historical_data_memory_ttl': 3600, 'historical_data_disk_ttl': 
        604800, 'recent_data_memory_ttl': 300, 'recent_data_disk_ttl': 
        86400, 'enable_parallel_processing': True, 'max_parallel_tasks': 8,
        'enable_metrics_collection': True, 'metrics_log_interval': 3600}

    @classmethod
    @with_exception_handling
    def load_config(cls, config_path: str=None) ->Dict[str, Any]:
        """
        Load caching configuration from a file or environment variables.
        
        Args:
            config_path: Optional path to a JSON config file
            
        Returns:
            Dictionary with configuration values
        """
        config = cls.DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                config.update(file_config)
            except Exception as e:
                print(f'Error loading cache config from {config_path}: {e}')
        for key in config.keys():
            env_key = f'FEATURE_STORE_CACHE_{key.upper()}'
            if env_key in os.environ:
                value = os.environ[env_key]
                if isinstance(config[key], bool):
                    config[key] = value.lower() in ('true', '1', 'yes', 'y',
                        'on')
                elif isinstance(config[key], int):
                    config[key] = int(value)
                elif isinstance(config[key], float):
                    config[key] = float(value)
                else:
                    config[key] = value
        if not os.path.isabs(config['disk_cache_path']):
            config['disk_cache_path'] = os.path.join(os.getcwd(), config[
                'disk_cache_path'])
        return config
