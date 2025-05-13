"""
Model Inference Cache Module.

This module provides caching functionality for ML model inference results.
It implements a decorator that can be applied to model inference methods
to cache their results based on input parameters.
"""

import functools
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Global cache storage
_model_prediction_cache = {}


def _create_cache_key(model_name: str, symbol: str, timeframe: str, features: Any) -> str:
    """
    Create a cache key for model inference.
    
    Args:
        model_name: Name of the model
        symbol: Trading symbol
        timeframe: Chart timeframe
        features: Features for prediction (can be DataFrame or dict)
        
    Returns:
        Cache key string
    """
    # Convert features to a hashable representation
    if isinstance(features, pd.DataFrame):
        # For DataFrames, use a hash of the values and column names
        features_hash = hashlib.md5(
            pd.util.hash_pandas_object(features).values.tobytes() + 
            str(features.columns.tolist()).encode()
        ).hexdigest()
    elif isinstance(features, dict):
        # For dictionaries, use a hash of the sorted JSON representation
        features_hash = hashlib.md5(
            json.dumps(features, sort_keys=True).encode()
        ).hexdigest()
    else:
        # For other types, use a hash of the string representation
        features_hash = hashlib.md5(str(features).encode()).hexdigest()
    
    # Combine all components into a single key
    return f"{model_name}:{symbol}:{timeframe}:{features_hash}"


def cache_model_inference(ttl: int = 1800):
    """
    Decorator for caching model inference results.
    
    Args:
        ttl: Time-to-live for cache entries in seconds (default: 30 minutes)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable):
    """
    Decorator.
    
    Args:
        func: Description of func
    
    """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

            # Extract parameters for cache key
            model_name = kwargs.get('model_name', args[1] if len(args) > 1 else None)
            symbol = kwargs.get('symbol', args[2] if len(args) > 2 else None)
            timeframe = kwargs.get('timeframe', args[3] if len(args) > 3 else None)
            features = kwargs.get('features', args[4] if len(args) > 4 else None)
            
            if not all([model_name, symbol, timeframe, features]):
                logger.warning("Missing required parameters for caching, skipping cache")
                return func(*args, **kwargs)
            
            # Create cache key
            cache_key = _create_cache_key(model_name, symbol, timeframe, features)
            
            # Check if result is in cache and not expired
            if cache_key in _model_prediction_cache:
                cache_entry = _model_prediction_cache[cache_key]
                if datetime.now() < cache_entry["expiry"]:
                    logger.debug(f"Cache hit for model {model_name}, symbol {symbol}, timeframe {timeframe}")
                    return cache_entry["result"]
            
            # Call the original function
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Store result in cache
            _model_prediction_cache[cache_key] = {
                "result": result,
                "expiry": datetime.now() + timedelta(seconds=ttl),
                "created": datetime.now(),
                "execution_time": execution_time
            }
            
            logger.debug(f"Cached model inference result for {model_name} (took {execution_time:.3f}s)")
            
            return result
        return wrapper
    return decorator


def clear_model_cache(model_name: Optional[str] = None, symbol: Optional[str] = None):
    """
    Clear the model inference cache.
    
    Args:
        model_name: Optional model name to clear only entries for that model
        symbol: Optional symbol to clear only entries for that symbol
    """
    global _model_prediction_cache
    
    if model_name is None and symbol is None:
        # Clear entire cache
        _model_prediction_cache = {}
        logger.info("Cleared entire model inference cache")
    else:
        # Clear specific entries
        keys_to_remove = []
        for key in _model_prediction_cache.keys():
            parts = key.split(':')
            if model_name and parts[0] == model_name:
                keys_to_remove.append(key)
            elif symbol and parts[1] == symbol:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del _model_prediction_cache[key]
        
        logger.info(f"Cleared {len(keys_to_remove)} entries from model inference cache")


def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the model inference cache.
    
    Returns:
        Dictionary with cache statistics
    """
    stats = {
        "total_entries": len(_model_prediction_cache),
        "active_entries": 0,
        "expired_entries": 0,
        "models": set(),
        "symbols": set(),
        "timeframes": set(),
        "avg_execution_time": 0.0
    }
    
    now = datetime.now()
    total_execution_time = 0.0
    
    for key, entry in _model_prediction_cache.items():
        parts = key.split(':')
        if len(parts) >= 3:
            stats["models"].add(parts[0])
            stats["symbols"].add(parts[1])
            stats["timeframes"].add(parts[2])
        
        if now < entry["expiry"]:
            stats["active_entries"] += 1
        else:
            stats["expired_entries"] += 1
        
        total_execution_time += entry.get("execution_time", 0.0)
    
    if stats["total_entries"] > 0:
        stats["avg_execution_time"] = total_execution_time / stats["total_entries"]
    
    # Convert sets to lists for JSON serialization
    stats["models"] = list(stats["models"])
    stats["symbols"] = list(stats["symbols"])
    stats["timeframes"] = list(stats["timeframes"])
    
    return stats
