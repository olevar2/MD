"""
Core Foundations: Unified Indicator Interface

This module provides a unified interface for accessing and calculating indicators
across different services, supporting incremental calculation and caching.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import hashlib
import json
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class CalculationMode(Enum):
    """Calculation modes for indicators"""
    FULL = auto()       # Full recalculation
    INCREMENTAL = auto() # Incremental calculation (append only)
    PARTIAL = auto()     # Partial recalculation (with specified range)


class CachePolicy(Enum):
    """Cache policies for indicator results"""
    NO_CACHE = auto()    # Do not cache results
    MEMORY = auto()      # Cache in memory
    PERSISTENT = auto()  # Cache in persistent storage (e.g., Redis)


class IndicatorResult:
    """Container for indicator calculation results with metadata"""
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 metadata: Dict[str, Any],
                 calculation_time: float,
                 cache_hit: bool = False):
        """
        Initialize the indicator result
        
        Args:
            data: DataFrame containing the calculated indicator values
            metadata: Dictionary with metadata about the calculation
            calculation_time: Time taken to calculate in seconds
            cache_hit: Whether the result was retrieved from cache
        """
        self.data = data
        self.metadata = metadata
        self.calculation_time = calculation_time
        self.cache_hit = cache_hit
        self.timestamp = datetime.now()
    
    def __repr__(self) -> str:
        """String representation of the indicator result"""
        return f"IndicatorResult(rows={len(self.data)}, cache_hit={self.cache_hit})"


class IndicatorCache:
    """Caching mechanism for indicator calculations"""
    
    def __init__(self, policy: CachePolicy = CachePolicy.MEMORY, max_size: int = 100):
        """
        Initialize the indicator cache
        
        Args:
            policy: Cache policy to use
            max_size: Maximum number of items to store in cache
        """
        self.policy = policy
        self.max_size = max_size
        self._cache: Dict[str, IndicatorResult] = {}
        self._hit_count = 0
        self._miss_count = 0
    
    def generate_key(self, indicator_name: str, params: Dict[str, Any], 
                    data_hash: str) -> str:
        """Generate a unique cache key based on indicator name, params, and data"""
        key_dict = {
            "indicator_name": indicator_name,
            "params": params,
            "data_hash": data_hash
        }
        key_str = json.dumps(key_dict, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[IndicatorResult]:
        """Get a result from the cache"""
        result = self._cache.get(key)
        if result:
            self._hit_count += 1
            return result
        self._miss_count += 1
        return None
    
    def put(self, key: str, result: IndicatorResult) -> None:
        """Store a result in the cache"""
        if self.policy == CachePolicy.NO_CACHE:
            return
            
        # Simple LRU-like eviction if cache gets too large
        if len(self._cache) >= self.max_size:
            # Remove oldest item (this is a simple approach)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            
        self._cache[key] = result
    
    def clear(self) -> None:
        """Clear the cache"""
        self._cache.clear()
        
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self._hit_count + self._miss_count
        hit_ratio = self._hit_count / total if total > 0 else 0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_ratio": hit_ratio
        }


class Indicator(ABC):
    """Abstract base class for all indicators"""
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the indicator
        
        Args:
            name: Name of the indicator
            description: Description of the indicator
        """
        self.name = name
        self.description = description
        self._last_result: Optional[IndicatorResult] = None
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate the indicator values
        
        Args:
            data: Input DataFrame with price/volume data
            **kwargs: Additional parameters for the calculation
            
        Returns:
            DataFrame with calculated indicator values
        """
        pass
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for this indicator"""
        return {}
    
    def get_required_columns(self) -> List[str]:
        """Get the columns required in the input data"""
        return ['open', 'high', 'low', 'close']
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that the input data has the required columns"""
        required = self.get_required_columns()
        return all(col in data.columns for col in required)


def implement_unified_interface():
    """
    Implements the unified API for indicators.
    - Develops cascaded and incremental calculation system.
    - Creates caching mechanism for repeated calculations.
    
    Returns:
        The indicator cache instance for global use
    """
    # Initialize the global cache for indicators
    global_cache = IndicatorCache(policy=CachePolicy.MEMORY, max_size=200)
    
    # Log that the unified interface has been initialized
    logger.info("Unified indicator interface initialized with memory caching")
    
    return global_cache


class IndicatorRegistry:
    """Registry for all available indicators"""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one registry exists"""
        if cls._instance is None:
            cls._instance = super(IndicatorRegistry, cls).__new__(cls)
            cls._instance._indicators = {}
            cls._instance._cache = implement_unified_interface()
        return cls._instance
    
    def register(self, indicator_class) -> None:
        """
        Register an indicator class
        
        Args:
            indicator_class: The indicator class to register
        """
        instance = indicator_class()
        self._indicators[instance.name] = indicator_class
        logger.debug(f"Registered indicator: {instance.name}")
    
    def get(self, name: str) -> Optional[type]:
        """
        Get an indicator class by name
        
        Args:
            name: Name of the indicator
            
        Returns:
            The indicator class or None if not found
        """
        return self._indicators.get(name)
    
    def list_all(self) -> List[str]:
        """List all registered indicators"""
        return list(self._indicators.keys())
    
    def calculate_indicator(self, 
                          name: str, 
                          data: pd.DataFrame,
                          mode: CalculationMode = CalculationMode.FULL,
                          use_cache: bool = True,
                          **params) -> IndicatorResult:
        """
        Calculate an indicator using the unified interface
        
        Args:
            name: Name of the indicator to calculate
            data: Input DataFrame
            mode: Calculation mode
            use_cache: Whether to use caching
            **params: Additional parameters for the indicator
            
        Returns:
            IndicatorResult with the calculation results
        """
        indicator_class = self.get(name)
        if not indicator_class:
            raise ValueError(f"Unknown indicator: {name}")
        
        # Create indicator instance
        indicator = indicator_class()
        
        # Handle caching if enabled
        if use_cache and mode == CalculationMode.FULL:
            # Generate data hash for cache key
            data_hash = hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()
            
            # Generate cache key
            cache_key = self._cache.generate_key(name, params, data_hash)
            
            # Try to get from cache
            cached_result = self._cache.get(cache_key)
            if cached_result:
                return cached_result
        
        # Calculate the indicator
        import time
        start_time = time.time()
        
        try:
            # Apply calculation mode
            if mode == CalculationMode.INCREMENTAL and indicator._last_result is not None:
                # Get the last timestamp from previous calculation
                last_timestamp = indicator._last_result.data.index[-1]
                
                # Filter for new data only
                new_data = data[data.index > last_timestamp]
                
                if len(new_data) > 0:
                    # Calculate only on new data
                    new_result = indicator.calculate(new_data, **params)
                    
                    # Concatenate with previous result
                    result_data = pd.concat([indicator._last_result.data, new_result])
                else:
                    # No new data, use previous result
                    result_data = indicator._last_result.data
            else:
                # Full calculation
                result_data = indicator.calculate(data, **params)
            
            # Create result object
            calculation_time = time.time() - start_time
            result = IndicatorResult(
                data=result_data,
                metadata={
                    "indicator": name,
                    "params": params,
                    "rows_processed": len(data)
                },
                calculation_time=calculation_time
            )
            
            # Store in cache if enabled
            if use_cache and mode == CalculationMode.FULL:
                self._cache.put(cache_key, result)
            
            # Store as last result for incremental calculations
            indicator._last_result = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating indicator {name}: {str(e)}")
            raise
    
    def calculate_multiple(self, 
                         indicators: List[Tuple[str, Dict[str, Any]]], 
                         data: pd.DataFrame,
                         mode: CalculationMode = CalculationMode.FULL,
                         use_cache: bool = True) -> Dict[str, IndicatorResult]:
        """
        Calculate multiple indicators in batch
        
        Args:
            indicators: List of (indicator_name, params) tuples
            data: Input DataFrame
            mode: Calculation mode
            use_cache: Whether to use caching
            
        Returns:
            Dictionary mapping indicator names to their results
        """
        results = {}
        
        for indicator_name, params in indicators:
            try:
                result = self.calculate_indicator(
                    indicator_name, data, mode, use_cache, **params
                )
                results[indicator_name] = result
            except Exception as e:
                logger.error(f"Error calculating {indicator_name}: {str(e)}")
                # Continue with other indicators even if one fails
                continue
                
        return results
    
    def compare_indicators(self, 
                         indicators: List[Tuple[str, Dict[str, Any]]], 
                         data: pd.DataFrame,
                         metrics: List[str] = None,
                         use_cache: bool = True) -> pd.DataFrame:
        """
        Compare multiple indicators using specified metrics
        
        Args:
            indicators: List of (indicator_name, params) tuples
            data: Input DataFrame
            metrics: List of metrics to use for comparison (e.g., 'correlation', 'lag', 'signal_quality')
            use_cache: Whether to use caching
            
        Returns:
            DataFrame with comparison results
        """
        if metrics is None:
            metrics = ['correlation', 'signal_agreement', 'calculation_time']
            
        # Get all indicator results
        results = self.calculate_multiple(indicators, data, use_cache=use_cache)
        
        # Prepare comparison DataFrame
        comparison = pd.DataFrame(index=[f"{name} {json.dumps(params)}" for name, params in indicators])
        
        # Calculate metrics
        for metric in metrics:
            if metric == 'correlation':
                # Extract main indicator column from each result and calculate correlation matrix
                indicator_series = {}
                for (name, params), result in zip(indicators, results.values()):
                    # Assuming the main indicator column has the same name as the indicator
                    if name in result.data.columns:
                        indicator_series[f"{name} {json.dumps(params)}"] = result.data[name]
                
                if indicator_series:
                    corr_matrix = pd.DataFrame(indicator_series).corr()
                    # Add correlation data to comparison DataFrame
                    comparison['avg_correlation'] = corr_matrix.mean(axis=1)
            
            elif metric == 'calculation_time':
                for (name, params), result in zip(indicators, results.values()):
                    comparison.at[f"{name} {json.dumps(params)}", 'calculation_time'] = result.calculation_time
            
            elif metric == 'signal_agreement':
                # This would require additional signal processing logic
                pass
        
        return comparison
    
    def optimize_parameters(self,
                          indicator_name: str,
                          data: pd.DataFrame,
                          param_grid: Dict[str, List[Any]],
                          target_column: str = 'close',
                          metric: str = 'correlation',
                          test_size: float = 0.3) -> Dict[str, Any]:
        """
        Find optimal parameters for an indicator using grid search
        
        Args:
            indicator_name: Name of the indicator
            data: Input DataFrame
            param_grid: Dictionary mapping parameter names to lists of values to test
            target_column: Column to use for optimization target
            metric: Metric to optimize ('correlation', 'profit', etc.)
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with optimal parameters and performance metrics
        """
        if not self.get(indicator_name):
            raise ValueError(f"Unknown indicator: {indicator_name}")
        
        # Prepare data - split into training and testing sets
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        # Generate all parameter combinations
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(itertools.product(*param_grid.values()))
        
        best_score = -float('inf')
        best_params = None
        results = []
        
        # Try each parameter combination
        for values in param_values:
            params = dict(zip(param_names, values))
            
            try:
                # Calculate indicator with current parameters
                result = self.calculate_indicator(
                    indicator_name, train_data, use_cache=False, **params
                )
                
                # Evaluate based on specified metric
                score = None
                if metric == 'correlation':
                    # Assuming the main indicator column has the same name as the indicator
                    # or is the first new column added by the indicator
                    output_columns = set(result.data.columns) - set(train_data.columns)
                    if output_columns:
                        indicator_col = list(output_columns)[0]
                        # Calculate correlation with target
                        correlation = result.data[indicator_col].corr(train_data[target_column])
                        score = abs(correlation)  # Use absolute correlation
                
                # Validate on test data
                test_result = self.calculate_indicator(
                    indicator_name, test_data, use_cache=False, **params
                )
                
                # Store result
                results.append({
                    'params': params,
                    'train_score': score,
                    'calculation_time': result.calculation_time,
                    'test_result': test_result
                })
                
                # Update best parameters
                if score is not None and score > best_score:
                    best_score = score
                    best_params = params
            
            except Exception as e:
                logger.warning(f"Parameter set {params} failed: {str(e)}")
                continue
        
        if best_params is None:
            logger.warning("No valid parameter combinations found")
            return {'success': False, 'error': 'No valid parameter combinations found'}
            
        return {
            'success': True,
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    @property
    def cache(self) -> IndicatorCache:
        """Get the indicator cache"""
        return self._cache
