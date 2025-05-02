"""
Error handling and performance optimization for advanced indicator integration.

This module provides utilities for robust error handling and performance
optimization when working with advanced indicators from the Analysis Engine.
"""

import logging
import time
from functools import wraps
from typing import Dict, Any, Optional, Callable, Type, Union, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class AdvancedIndicatorError(Exception):
    """Error raised during advanced indicator calculation."""
    pass


def graceful_fallback(fallback_value=None, log_error=True):
    """
    Decorator that provides graceful fallback for advanced indicator functions.
    
    Args:
        fallback_value: Value to return if an error occurs (None, empty DataFrame, etc.)
        log_error: Whether to log the error
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                
                # Determine the appropriate fallback value
                if fallback_value is not None:
                    return fallback_value
                    
                # If first argument is a DataFrame, return it as fallback
                elif args and isinstance(args[0], pd.DataFrame):
                    return args[0].copy()
                    
                # Default fallback for different return types
                elif hasattr(func, '__annotations__') and 'return' in func.__annotations__:
                    return_type = func.__annotations__['return']
                    if return_type == pd.DataFrame or 'DataFrame' in str(return_type):
                        return pd.DataFrame()
                    elif return_type == Dict:
                        return {}
                    elif return_type == List:
                        return []
                        
                # Ultimate default fallback
                return None
                
        return wrapper
    return decorator


def performance_tracking(threshold_ms=100):
    """
    Decorator that tracks performance of advanced indicator calculations.
    
    Args:
        threshold_ms: Log a warning if execution time exceeds this threshold
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Log if execution time exceeds threshold
            if execution_time > threshold_ms:
                logger.warning(
                    f"Performance warning: {func.__name__} took {execution_time:.2f}ms "
                    f"(threshold: {threshold_ms}ms)"
                )
                
            return result
            
        return wrapper
    return decorator


class AdvancedIndicatorOptimizer:
    """
    Optimizer for advanced indicators to improve calculation performance.
    
    This class provides methods to optimize the calculation of advanced
    indicators by reusing intermediate results, detecting redundant calculations,
    and applying other performance optimizations.
    """
    
    def __init__(self):
        """Initialize the optimizer."""
        self.calculation_cache = {}
        self.intermediate_results = {}
        
    def optimize_calculation(
        self,
        data: pd.DataFrame,
        indicator_name: str,
        params: Dict[str, Any],
        calculation_func: Callable
    ) -> pd.DataFrame:
        """
        Optimize the calculation of an advanced indicator.
        
        Args:
            data: Input DataFrame
            indicator_name: Name of the indicator
            params: Parameters for the indicator
            calculation_func: Function to calculate the indicator
            
        Returns:
            DataFrame with indicator results
        """
        # Create cache key from indicator name, params hash, and data range
        cache_key = self._create_cache_key(indicator_name, params, data)
        
        # Check if we have cached results
        if cache_key in self.calculation_cache:
            cached_result = self.calculation_cache[cache_key]
            
            # Verify the cached result matches our data index
            if cached_result.index.equals(data.index):
                return cached_result
                
        # Perform the calculation
        with self._track_time(indicator_name, params):
            result = calculation_func(data)
            
        # Cache the result
        self.calculation_cache[cache_key] = result
        
        return result
        
    def _create_cache_key(
        self,
        indicator_name: str,
        params: Dict[str, Any],
        data: pd.DataFrame
    ) -> str:
        """Create a unique cache key for the calculation."""
        # Use string representation of params sorted by key for consistency
        params_str = str(sorted(params.items()))
        
        # Include data range information in the key
        date_range = f"{data.index.min()}_{data.index.max()}_{len(data)}"
        
        return f"{indicator_name}_{params_str}_{date_range}"
        
    def _track_time(self, indicator_name: str, params: Dict[str, Any]):
        """Context manager to track calculation time."""
        class TimeTracker:
            def __init__(self, optimizer, indicator_name, params):
                self.optimizer = optimizer
                self.indicator_name = indicator_name
                self.params = params
                self.start_time = None
                
            def __enter__(self):
                self.start_time = time.time()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                execution_time = (time.time() - self.start_time) * 1000  # Convert to ms
                logger.debug(
                    f"Calculated {self.indicator_name} in {execution_time:.2f}ms "
                    f"with params {self.params}"
                )
                
        return TimeTracker(self, indicator_name, params)
        
    def clear_cache(self):
        """Clear the calculation cache."""
        self.calculation_cache.clear()
        self.intermediate_results.clear()


# Global optimizer instance
optimizer = AdvancedIndicatorOptimizer()
