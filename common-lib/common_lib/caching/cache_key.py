"""
Cache Key Module for Forex Trading Platform.

This module provides utilities for generating and managing cache keys.
"""

import hashlib
import json
from typing import Dict, Any, Optional, Union, Callable, List, Tuple
from datetime import datetime

class CacheKey:
    """
    Class for generating and managing cache keys.
    
    This class provides methods for generating cache keys based on various
    parameters, including function calls, data parameters, and timestamps.
    """
    
    @staticmethod
    def from_function(func: Callable, *args, **kwargs) -> str:
        """
        Generate a cache key from a function call.
        
        Args:
            func: Function
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cache key
        """
        # Get function name and module
        func_name = func.__name__
        module_name = func.__module__
        
        # Convert args and kwargs to a string
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        
        # Generate a hash
        key_str = f"{module_name}.{func_name}:{args_str}:{kwargs_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @staticmethod
    def from_params(
        prefix: str,
        params: Dict[str, Any],
        include_keys: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None
    ) -> str:
        """
        Generate a cache key from parameters.
        
        Args:
            prefix: Prefix for the key
            params: Parameters
            include_keys: Keys to include (if None, include all)
            exclude_keys: Keys to exclude
            
        Returns:
            Cache key
        """
        # Filter parameters
        filtered_params = {}
        
        if include_keys:
            # Include only specified keys
            for key in include_keys:
                if key in params:
                    filtered_params[key] = params[key]
        else:
            # Include all keys except excluded
            filtered_params = params.copy()
            
            if exclude_keys:
                for key in exclude_keys:
                    if key in filtered_params:
                        del filtered_params[key]
        
        # Sort parameters by key
        sorted_params = {k: filtered_params[k] for k in sorted(filtered_params.keys())}
        
        # Convert to JSON string
        params_str = json.dumps(sorted_params, sort_keys=True, default=str)
        
        # Generate a hash
        key_str = f"{prefix}:{params_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @staticmethod
    def from_timestamp(
        prefix: str,
        timestamp: Union[datetime, str],
        granularity: str = "minute"
    ) -> str:
        """
        Generate a cache key from a timestamp with specified granularity.
        
        Args:
            prefix: Prefix for the key
            timestamp: Timestamp
            granularity: Granularity (second, minute, hour, day)
            
        Returns:
            Cache key
        """
        # Convert timestamp to datetime if needed
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        
        # Format timestamp based on granularity
        if granularity == "second":
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        elif granularity == "minute":
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:00")
        elif granularity == "hour":
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:00:00")
        elif granularity == "day":
            timestamp_str = timestamp.strftime("%Y-%m-%d 00:00:00")
        else:
            raise ValueError(f"Invalid granularity: {granularity}")
        
        # Generate a hash
        key_str = f"{prefix}:{timestamp_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @staticmethod
    def from_version(
        prefix: str,
        data_key: str,
        version: Union[int, str]
    ) -> str:
        """
        Generate a cache key with version information.
        
        Args:
            prefix: Prefix for the key
            data_key: Data key
            version: Version
            
        Returns:
            Cache key
        """
        # Generate a hash
        key_str = f"{prefix}:{data_key}:v{version}"
        return hashlib.md5(key_str.encode()).hexdigest()

def generate_cache_key(
    prefix: str,
    *args,
    **kwargs
) -> str:
    """
    Generate a cache key from arguments.
    
    Args:
        prefix: Prefix for the key
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Cache key
    """
    # Convert args and kwargs to a string
    args_str = str(args)
    kwargs_str = str(sorted(kwargs.items()))
    
    # Generate a hash
    key_str = f"{prefix}:{args_str}:{kwargs_str}"
    return hashlib.md5(key_str.encode()).hexdigest()
