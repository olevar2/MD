"""
Configuration validation for the caching system.

This module contains functions for validating cache configuration parameters
and applying sensible defaults.
"""
from typing import Dict, Any
import os
import json
import logging
logger = logging.getLogger(__name__)


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

def validate_config(config: Dict[str, Any]) ->Dict[str, Any]:
    """
    Validate cache configuration and apply defaults for missing values.
    
    Args:
        config: The configuration dictionary to validate
        
    Returns:
        Dict[str, Any]: A validated configuration dictionary with defaults applied
    """
    validated_config = {}
    validated_config['memory_cache_size'] = config.get('memory_cache_size',
        1073741824)
    validated_config['memory_cache_ttl'] = config_manager.get('memory_cache_ttl', 300)
    validated_config['use_disk_cache'] = config_manager.get('use_disk_cache', True)
    if validated_config['use_disk_cache']:
        default_cache_dir = os.path.join(os.getcwd(), 'cache')
        validated_config['disk_cache_path'] = config.get('disk_cache_path',
            default_cache_dir)
        os.makedirs(validated_config['disk_cache_path'], exist_ok=True)
        validated_config['disk_cache_size'] = config.get('disk_cache_size',
            53687091200)
        validated_config['disk_cache_ttl'] = config.get('disk_cache_ttl', 86400
            )
    validated_config['historical_data_memory_ttl'] = config.get(
        'historical_data_memory_ttl', 3600)
    validated_config['historical_data_disk_ttl'] = config.get(
        'historical_data_disk_ttl', 604800)
    validated_config['recent_data_memory_ttl'] = config.get(
        'recent_data_memory_ttl', 300)
    validated_config['recent_data_disk_ttl'] = config.get(
        'recent_data_disk_ttl', 86400)
    validated_config['enable_parallel_processing'] = config.get(
        'enable_parallel_processing', True)
    validated_config['max_parallel_tasks'] = config.get('max_parallel_tasks', 8
        )
    validated_config['enable_metrics_collection'] = config.get(
        'enable_metrics_collection', True)
    validated_config['metrics_log_interval'] = config.get(
        'metrics_log_interval', 3600)
    validated_config['metrics_retention_days'] = config.get(
        'metrics_retention_days', 30)
    if validated_config['memory_cache_size'] < 1048576:
        logger.warning(
            'Memory cache size is very small (<1MB), this may affect performance'
            )
    if validated_config['use_disk_cache'] and validated_config[
        'disk_cache_size'] < 1073741824:
        logger.warning(
            'Disk cache size is very small (<1GB), this may affect performance'
            )
    return validated_config


@with_exception_handling
def load_config_file(config_path: str) ->Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the JSON configuration file
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return validate_config(config)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(
            f'Failed to load cache configuration from {config_path}: {str(e)}')
        logger.info('Using default cache configuration')
        return validate_config({})
