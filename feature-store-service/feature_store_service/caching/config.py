"""
Configuration module for the Feature Store Caching system.

This module provides configuration options and loading functionality
for the multi-tiered caching system.
"""
import os
import json
from typing import Dict, Any


class CacheConfig:
    """Configuration handler for the caching system."""
    
    DEFAULT_CONFIG = {
        # Memory cache settings
        'memory_cache_size': 1_000_000_000,  # 1GB
        'memory_cache_ttl': 300,  # 5 minutes
        
        # Disk cache settings
        'use_disk_cache': True,
        'disk_cache_path': 'cache',  # Will be resolved relative to the service root
        'disk_cache_size': 50_000_000_000,  # 50GB
        'disk_cache_ttl': 86400,  # 24 hours
        
        # Cache behavior settings
        'historical_data_memory_ttl': 3600,  # 1 hour for historical data in memory
        'historical_data_disk_ttl': 604800,  # 1 week for historical data on disk
        'recent_data_memory_ttl': 300,  # 5 minutes for recent data in memory
        'recent_data_disk_ttl': 86400,  # 24 hours for recent data on disk
        
        # Performance tuning
        'enable_parallel_processing': True,
        'max_parallel_tasks': 8,
        'enable_metrics_collection': True,
        'metrics_log_interval': 3600,  # Log metrics every hour
    }
    
    @classmethod
    def load_config(cls, config_path: str = None) -> Dict[str, Any]:
        """
        Load caching configuration from a file or environment variables.
        
        Args:
            config_path: Optional path to a JSON config file
            
        Returns:
            Dictionary with configuration values
        """
        # Start with default config
        config = cls.DEFAULT_CONFIG.copy()
        
        # Override from config file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                config.update(file_config)
            except Exception as e:
                print(f"Error loading cache config from {config_path}: {e}")
        
        # Override from environment variables
        for key in config.keys():
            env_key = f"FEATURE_STORE_CACHE_{key.upper()}"
            if env_key in os.environ:
                value = os.environ[env_key]
                
                # Convert value to the right type
                if isinstance(config[key], bool):
                    config[key] = value.lower() in ('true', '1', 'yes', 'y', 'on')
                elif isinstance(config[key], int):
                    config[key] = int(value)
                elif isinstance(config[key], float):
                    config[key] = float(value)
                else:
                    config[key] = value
        
        # Make disk_cache_path absolute if not already
        if not os.path.isabs(config['disk_cache_path']):
            config['disk_cache_path'] = os.path.join(
                os.getcwd(), 
                config['disk_cache_path']
            )
        
        return config
