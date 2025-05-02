"""
Indicator Interface Module

This module provides a unified interface for technical indicators with advanced functionality:
- Registry system for indicator management
- Caching mechanism for efficient repeated calculations
- Support for incremental and partial calculation modes
- Advanced calculation capabilities including batch processing and optimization
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Callable, Optional, Union
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from dataclasses import dataclass


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CalculationMetrics:
    """Stores metrics about indicator calculations."""
    calculation_time: float
    cache_hit: bool
    data_size: int
    parameters: Dict[str, Any]


class CalculationMode:
    """Enumeration of supported calculation modes."""
    STANDARD = "standard"  # Calculate for entire dataset
    INCREMENTAL = "incremental"  # Calculate only for new data points
    PARTIAL = "partial"  # Calculate for specific window/range
    OPTIMIZED = "optimized"  # Use optimized algorithm when available


class Indicator(ABC):
    """Abstract base class for all indicators."""
    
    def __init__(self, name: str, description: str, category: str = "general"):
        """
        Initialize indicator with metadata.
        
        Args:
            name: Unique identifier for the indicator
            description: Brief description of what the indicator measures
            category: Classification category (e.g., trend, momentum, volatility)
        """
        self.name = name
        self.description = description
        self.category = category
        self.calculation_metrics = None
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **parameters) -> pd.DataFrame:
        """
        Calculate indicator values.
        
        Args:
            data: Input price/volume data
            **parameters: Indicator-specific parameters
            
        Returns:
            DataFrame with indicator values
        """
        pass
    
    def validate_parameters(self, **parameters) -> bool:
        """
        Validate input parameters.
        
        Args:
            **parameters: Parameters to validate
            
        Returns:
            True if parameters are valid, raises ValueError otherwise
        """
        # Default implementation - override in derived classes
        return True
    
    def generate_signals(self, data: pd.DataFrame, indicator_values: pd.DataFrame, **parameters) -> pd.DataFrame:
        """
        Generate trading signals based on indicator values.
        
        Args:
            data: Original price/volume data
            indicator_values: Calculated indicator values
            **parameters: Signal generation parameters
            
        Returns:
            DataFrame with signals (1 for buy, -1 for sell, 0 for neutral)
        """
        # Default implementation - override in derived classes
        return pd.DataFrame(0, index=data.index, columns=["signal"])
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description} ({self.category})"


class IndicatorRegistry:
    """Central registry and management system for all technical indicators."""
    
    def __init__(self):
        """Initialize the registry."""
        self._indicators: Dict[str, Indicator] = {}
        self._cache = {}
        self._cache_stats = {"hits": 0, "misses": 0}
        self.max_cache_size = 100
    
    def register(self, indicator: Indicator) -> None:
        """
        Register a new indicator.
        
        Args:
            indicator: The indicator instance to register
        """
        if indicator.name in self._indicators:
            logger.warning(f"Indicator '{indicator.name}' already exists and will be overwritten")
        self._indicators[indicator.name] = indicator
        logger.info(f"Registered indicator: {indicator.name}")
    
    def unregister(self, indicator_name: str) -> None:
        """
        Remove an indicator from the registry.
        
        Args:
            indicator_name: Name of the indicator to remove
        """
        if indicator_name in self._indicators:
            del self._indicators[indicator_name]
            logger.info(f"Unregistered indicator: {indicator_name}")
        else:
            logger.warning(f"Cannot unregister non-existent indicator: {indicator_name}")
    
    def get_indicator(self, indicator_name: str) -> Indicator:
        """
        Get an indicator by name.
        
        Args:
            indicator_name: Name of the indicator to retrieve
            
        Returns:
            The requested indicator instance
        """
        if indicator_name not in self._indicators:
            raise KeyError(f"Indicator not found: {indicator_name}")
        return self._indicators[indicator_name]
    
    def list_indicators(self, category: Optional[str] = None) -> List[str]:
        """
        List available indicators, optionally filtered by category.
        
        Args:
            category: Filter by indicator category
            
        Returns:
            List of indicator names
        """
        if category:
            return [name for name, ind in self._indicators.items() if ind.category == category]
        return list(self._indicators.keys())
    
    def _generate_cache_key(self, indicator_name: str, data_hash: str, parameters: Dict[str, Any]) -> str:
        """Generate a unique cache key based on inputs."""
        # Convert parameters dict to sorted tuple for consistent hashing
        param_str = str(sorted([(k, v) for k, v in parameters.items()]))
        return f"{indicator_name}_{data_hash}_{param_str}"
    
    def _compute_data_hash(self, data: pd.DataFrame) -> str:
        """Compute a hash for the input data."""
        # Using a simple hash of data shape and last few values for efficiency
        shape_hash = f"{data.shape}"
        data_sample = data.iloc[-5:] if len(data) >= 5 else data
        value_hash = str(hash(tuple(data_sample.values.flatten())))
        return f"{shape_hash}_{value_hash}"
    
    def calculate_indicator(self, 
                           indicator_name: str, 
                           data: pd.DataFrame, 
                           mode: str = CalculationMode.STANDARD,
                           use_cache: bool = True,
                           **parameters) -> pd.DataFrame:
        """
        Calculate indicator values with optional caching.
        
        Args:
            indicator_name: Name of the indicator to calculate
            data: Input price/volume data
            mode: Calculation mode (standard, incremental, partial, optimized)
            use_cache: Whether to use cached results when available
            **parameters: Indicator-specific parameters
            
        Returns:
            DataFrame with calculated indicator values
        """
        indicator = self.get_indicator(indicator_name)
        indicator.validate_parameters(**parameters)
        
        # Handle caching
        data_hash = self._compute_data_hash(data)
        cache_hit = False
        
        if use_cache:
            cache_key = self._generate_cache_key(indicator_name, data_hash, parameters)
            if cache_key in self._cache:
                self._cache_stats["hits"] += 1
                logger.debug(f"Cache hit for indicator {indicator_name}")
                cache_hit = True
                result = self._cache[cache_key]
            else:
                self._cache_stats["misses"] += 1
                logger.debug(f"Cache miss for indicator {indicator_name}")
        
        if not use_cache or not cache_hit:
            # Measure calculation time
            start_time = time.time()
            
            # Handle different calculation modes
            if mode == CalculationMode.STANDARD:
                result = indicator.calculate(data, **parameters)
            elif mode == CalculationMode.INCREMENTAL:
                # Only calculate for new data points if cached result exists
                if cache_hit and hasattr(indicator, 'calculate_incremental'):
                    last_idx = self._cache[cache_key].index[-1]
                    new_data = data[data.index > last_idx]
                    incremental_result = indicator.calculate_incremental(data, new_data, **parameters)
                    result = pd.concat([self._cache[cache_key], incremental_result])
                else:
                    result = indicator.calculate(data, **parameters)
            elif mode == CalculationMode.PARTIAL:
                if 'start_idx' in parameters and 'end_idx' in parameters:
                    partial_data = data.loc[parameters['start_idx']:parameters['end_idx']]
                    result = indicator.calculate(partial_data, **parameters)
                else:
                    result = indicator.calculate(data, **parameters)
            elif mode == CalculationMode.OPTIMIZED:
                if hasattr(indicator, 'calculate_optimized'):
                    result = indicator.calculate_optimized(data, **parameters)
                else:
                    result = indicator.calculate(data, **parameters)
            else:
                raise ValueError(f"Unsupported calculation mode: {mode}")
            
            calculation_time = time.time() - start_time
            
            # Store calculation metrics
            indicator.calculation_metrics = CalculationMetrics(
                calculation_time=calculation_time,
                cache_hit=cache_hit,
                data_size=len(data),
                parameters=parameters
            )
            
            # Update cache
            if use_cache:
                # Manage cache size
                if len(self._cache) >= self.max_cache_size:
                    # Simple LRU implementation - remove a random item
                    # In production, would use a more sophisticated LRU strategy
                    self._cache.pop(next(iter(self._cache)))
                
                cache_key = self._generate_cache_key(indicator_name, data_hash, parameters)
                self._cache[cache_key] = result
        
        return result
    
    def batch_calculate(self, 
                        indicator_names: List[str], 
                        data: pd.DataFrame, 
                        parallel: bool = True,
                        **parameters_dict) -> Dict[str, pd.DataFrame]:
        """
        Calculate multiple indicators in batch, optionally in parallel.
        
        Args:
            indicator_names: List of indicator names to calculate
            data: Input price/volume data
            parallel: Whether to calculate in parallel
            **parameters_dict: Dictionary mapping indicator names to their parameters
            
        Returns:
            Dictionary mapping indicator names to their calculated values
        """
        results = {}
        
        if parallel:
            # Calculate indicators in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                futures = {}
                for name in indicator_names:
                    params = parameters_dict.get(name, {})
                    futures[name] = executor.submit(
                        self.calculate_indicator, name, data, **params
                    )
                
                # Collect results
                for name, future in futures.items():
                    try:
                        results[name] = future.result()
                    except Exception as e:
                        logger.error(f"Error calculating indicator {name}: {str(e)}")
                        results[name] = None
        else:
            # Calculate indicators sequentially
            for name in indicator_names:
                try:
                    params = parameters_dict.get(name, {})
                    results[name] = self.calculate_indicator(name, data, **params)
                except Exception as e:
                    logger.error(f"Error calculating indicator {name}: {str(e)}")
                    results[name] = None
        
        return results
    
    def compare_indicators(self, 
                          indicator_names: List[str], 
                          data: pd.DataFrame, 
                          comparison_metric: Callable[[pd.DataFrame], float] = None,
                          **parameters_dict) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple indicators based on a metric function.
        
        Args:
            indicator_names: List of indicator names to compare
            data: Input price/volume data
            comparison_metric: Function to calculate comparison metric
                              If None, uses calculation time as the metric
            **parameters_dict: Dictionary mapping indicator names to their parameters
            
        Returns:
            Dictionary with comparison results
        """
        if comparison_metric is None:
            # Default comparison is based on calculation time
            comparison_metric = lambda df, ind: ind.calculation_metrics.calculation_time if ind.calculation_metrics else float('inf')
        
        # Calculate all indicators
        results = {}
        indicator_values = {}
        
        for name in indicator_names:
            indicator = self.get_indicator(name)
            params = parameters_dict.get(name, {})
            
            # Calculate with cache disabled to get accurate performance metrics
            indicator_values[name] = self.calculate_indicator(name, data, use_cache=False, **params)
            
            # Store results
            results[name] = {
                "calculation_time": indicator.calculation_metrics.calculation_time,
                "data_points": len(indicator_values[name].dropna()),
                "parameters": params
            }
            
            # Apply custom metric if provided
            if comparison_metric:
                results[name]["metric_value"] = comparison_metric(indicator_values[name], indicator)
        
        # Add ranking based on metric
        if all("metric_value" in result for result in results.values()):
            sorted_names = sorted(results.keys(), key=lambda name: results[name]["metric_value"])
            for i, name in enumerate(sorted_names):
                results[name]["rank"] = i + 1
        
        return results
    
    def optimize_parameters(self, 
                           indicator_name: str, 
                           data: pd.DataFrame, 
                           param_grid: Dict[str, List[Any]],
                           objective_function: Callable[[pd.DataFrame, pd.DataFrame], float],
                           mode: str = "maximize",
                           max_iterations: int = 10,
                           parallel: bool = True) -> Tuple[Dict[str, Any], float]:
        """
        Find optimal parameters for an indicator using grid search.
        
        Args:
            indicator_name: Name of the indicator to optimize
            data: Input price/volume data
            param_grid: Dictionary mapping parameter names to lists of values to try
            objective_function: Function that evaluates indicator performance
                              Takes (original_data, indicator_values) and returns a scalar
            mode: "maximize" or "minimize" the objective function
            max_iterations: Maximum number of parameter combinations to try
            parallel: Whether to run evaluations in parallel
            
        Returns:
            Tuple of (best_parameters, best_score)
        """
        # Generate parameter combinations
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(itertools.product(*(param_grid[name] for name in param_names)))
        
        # Limit iterations if needed
        if len(param_values) > max_iterations:
            logger.warning(f"Parameter grid has {len(param_values)} combinations, limiting to {max_iterations}")
            # Simple random sampling - could be enhanced with smarter search strategies
            import random
            param_values = random.sample(param_values, max_iterations)
        
        best_score = float('-inf') if mode == "maximize" else float('inf')
        best_params = None
        
        def evaluate_params(params_tuple):
            params_dict = {name: value for name, value in zip(param_names, params_tuple)}
            try:
                # Calculate indicator with these parameters
                indicator_values = self.calculate_indicator(indicator_name, data, **params_dict)
                # Evaluate objective function
                score = objective_function(data, indicator_values)
                return params_dict, score
            except Exception as e:
                logger.error(f"Error optimizing with parameters {params_dict}: {str(e)}")
                return params_dict, float('-inf') if mode == "maximize" else float('inf')
        
        if parallel:
            # Run evaluations in parallel
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(evaluate_params, param_values))
        else:
            # Run evaluations sequentially
            results = [evaluate_params(params) for params in param_values]
        
        # Find best parameters
        for params, score in results:
            if (mode == "maximize" and score > best_score) or \
               (mode == "minimize" and score < best_score):
                best_score = score
                best_params = params
        
        return best_params, best_score
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cache performance."""
        return self._cache_stats.copy()
    
    def clear_cache(self) -> None:
        """Clear the calculation cache."""
        cache_size = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared indicator cache ({cache_size} entries)")


# Create a global registry instance
indicator_registry = IndicatorRegistry()
