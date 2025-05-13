"""
Configuration Loader

This module provides utilities for loading configuration from various sources
with platform-specific defaults.
"""
import os
import sys
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
from analysis_engine.utils.platform_compatibility import PlatformInfo, PlatformCompatibility
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class ConfigLoader:
    """Configuration loader with platform-specific defaults."""

    def __init__(self, app_name: str='forex-platform', config_name: str=
        'config', config_format: str='yaml'):
        """
        Initialize the configuration loader.
        
        Args:
            app_name: Application name
            config_name: Configuration file name (without extension)
            config_format: Configuration file format ("yaml" or "json")
        """
        self.app_name = app_name
        self.config_name = config_name
        self.config_format = config_format
        self.platform_info = PlatformInfo.get_platform_info()
        self.config_dir = PlatformCompatibility.get_config_dir()
        self.data_dir = PlatformCompatibility.get_data_dir()
        self.cache_dir = PlatformCompatibility.get_cache_dir()
        PlatformCompatibility.ensure_dir_exists(self.config_dir)
        PlatformCompatibility.ensure_dir_exists(self.data_dir)
        PlatformCompatibility.ensure_dir_exists(self.cache_dir)
        self.config = {}
        self.load_config()

    @with_resilience('get_config_paths')
    def get_config_paths(self) ->List[str]:
        """
        Get configuration file paths in order of precedence.
        
        Returns:
            List of configuration file paths
        """
        ext = '.yaml' if self.config_format == 'yaml' else '.json'
        current_dir = os.getcwd()
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.
            path.abspath(__file__))))
        paths = [os.path.join(current_dir, f'{self.config_name}{ext}'), os.
            path.join(self.config_dir, f'{self.config_name}{ext}'), os.path
            .join(script_dir, f'{self.config_name}{ext}'), self.
            _get_system_config_path(ext)]
        return paths

    def _get_system_config_path(self, ext: str) ->str:
        """
        Get system-wide configuration path.
        
        Args:
            ext: File extension
            
        Returns:
            System-wide configuration path
        """
        os_name = self.platform_info['os']
        if os_name == 'windows':
            return os.path.join(os.environ.get('PROGRAMDATA',
                'C:\\ProgramData'), self.app_name, f'{self.config_name}{ext}')
        elif os_name == 'macos':
            return (
                f'/Library/Application Support/{self.app_name}/{self.config_name}{ext}'
                )
        else:
            return f'/etc/{self.app_name}/{self.config_name}{ext}'

    @with_database_resilience('load_config')
    @with_exception_handling
    def load_config(self) ->Dict[str, Any]:
        """
        Load configuration from all sources.
        
        Returns:
            Merged configuration
        """
        self.config = self.get_default_config()
        paths = self.get_config_paths()
        for path in paths:
            if os.path.exists(path):
                logger.info(f'Loading configuration from {path}')
                try:
                    config = self._load_file(path)
                    self._merge_config(config)
                except Exception as e:
                    logger.error(
                        f'Error loading configuration from {path}: {e}')
        self._load_env_vars()
        self._apply_platform_overrides()
        return self.config

    def _load_file(self, path: str) ->Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            path: File path
            
        Returns:
            Configuration dictionary
        """
        with open(path, 'r') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                return yaml.safe_load(f) or {}
            elif path.endswith('.json'):
                return json.load(f) or {}
            else:
                raise ValueError(f'Unsupported file format: {path}')

    def _merge_config(self, config: Dict[str, Any]) ->None:
        """
        Merge configuration into the current configuration.
        
        Args:
            config: Configuration to merge
        """
        self._merge_dicts(self.config, config)

    def _merge_dicts(self, target: Dict[str, Any], source: Dict[str, Any]
        ) ->None:
        """
        Recursively merge dictionaries.
        
        Args:
            target: Target dictionary
            source: Source dictionary
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(
                value, dict):
                self._merge_dicts(target[key], value)
            else:
                target[key] = value

    def _load_env_vars(self) ->None:
        """Load configuration from environment variables."""
        prefix = f"{self.app_name.upper().replace('-', '_')}_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                keys = config_key.split('_')
                if value.lower() in ('true', 'yes', '1'):
                    value = True
                elif value.lower() in ('false', 'no', '0'):
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.'
                    ) == 1:
                    value = float(value)
                self._set_nested_value(self.config, keys, value)

    def _set_nested_value(self, config: Dict[str, Any], keys: List[str],
        value: Any) ->None:
        """
        Set a nested value in the configuration.
        
        Args:
            config: Configuration dictionary
            keys: List of nested keys
            value: Value to set
        """
        if len(keys) == 1:
            config[keys[0]] = value
        else:
            if keys[0] not in config:
                config[keys[0]] = {}
            self._set_nested_value(config[keys[0]], keys[1:], value)

    def _apply_platform_overrides(self) ->None:
        """Apply platform-specific overrides."""
        os_name = self.platform_info['os']
        platform_section = f'platform_{os_name}'
        if platform_section in self.config:
            platform_config = self.config[platform_section]
            self._merge_config(platform_config)
            for key in list(self.config.keys()):
                if key.startswith('platform_'):
                    del self.config[key]

    @with_resilience('get_default_config')
    def get_default_config(self) ->Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Default configuration
        """
        os_name = self.platform_info['os']
        has_gpu = self.platform_info['has_gpu']
        cpu_cores = self.platform_info['cpu']['logical_cores']
        optimal_threads = PlatformCompatibility.get_optimal_thread_count()
        optimal_batch_size = PlatformCompatibility.get_optimal_batch_size()
        optimal_memory_limit = PlatformCompatibility.get_optimal_memory_limit()
        return {'service': {'name': 'analysis-engine', 'host': '0.0.0.0',
            'port': 8000, 'debug': False, 'log_level': 'INFO'}, 'cache': {
            'default_ttl_seconds': 3600, 'max_size': 10000,
            'cleanup_interval_seconds': 300, 'adaptive_ttl': True,
            'memory_limit_bytes': optimal_memory_limit, 'cache_dir': self.
            cache_dir}, 'parallel_processing': {'min_workers': max(1, 
            optimal_threads // 2), 'max_workers': optimal_threads,
            'task_timeout_seconds': 30}, 'tracing': {'enable_tracing': True,
            'sampling_rate': 0.1, 'otlp_endpoint': 'http://localhost:4317'},
            'gpu': {'enable_gpu': has_gpu, 'memory_limit_mb': 1024,
            'batch_size': optimal_batch_size}, 'predictive_cache': {
            'prediction_threshold': 0.7, 'max_precompute_workers': max(1, 
            optimal_threads // 4), 'precomputation_interval_seconds': 10,
            'pattern_history_size': 1000}, 'data': {'data_dir': self.
            data_dir}, 'ml': {'model_dir': os.path.join(self.data_dir,
            'models')}, 'platform_windows': {'service': {'host':
            'localhost'}, 'tracing': {'otlp_endpoint':
            'http://localhost:4317'}}, 'platform_linux': {'tracing': {
            'otlp_endpoint': 'http://jaeger-collector:4317'}},
            'platform_macos': {'service': {'host': 'localhost'}, 'tracing':
            {'otlp_endpoint': 'http://localhost:4317'}}}

    def get(self, key: str, default: Any=None) ->Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (dot-separated for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) ->None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (dot-separated for nested keys)
            value: Value to set
        """
        keys = key.split('.')
        self._set_nested_value(self.config, keys, value)

    def save(self, path: Optional[str]=None) ->None:
        """
        Save configuration to a file.
        
        Args:
            path: File path (if None, use user config path)
        """
        if path is None:
            ext = '.yaml' if self.config_format == 'yaml' else '.json'
            path = os.path.join(self.config_dir, f'{self.config_name}{ext}')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                yaml.dump(self.config, f, default_flow_style=False)
            elif path.endswith('.json'):
                json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f'Unsupported file format: {path}')
        logger.info(f'Configuration saved to {path}')

    def reset(self) ->None:
        """Reset configuration to defaults."""
        self.config = self.get_default_config()

    def __getitem__(self, key: str) ->Any:
        """Get a configuration value."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) ->None:
        """Set a configuration value."""
        self.set(key, value)

    def __contains__(self, key: str) ->bool:
        """Check if a configuration key exists."""
        return self.get(key) is not None
