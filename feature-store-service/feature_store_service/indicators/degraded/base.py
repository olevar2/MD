"""
Base Classes for Degraded Mode Indicators

This module provides base classes and decorators for implementing
degraded mode versions of technical indicators that optimize for
performance at the cost of some accuracy.
"""

from abc import ABC, abstractmethod
import functools
import logging
from typing import Any, Dict, Callable, Optional, Type, Union
import pandas as pd
import numpy as np

from ..base_indicator import BaseIndicator


class DegradedModeIndicator(ABC):
    """
    Base class for all degraded mode indicators.
    
    Degraded mode indicators provide simplified calculations that are
    more efficient but may have reduced accuracy compared to their
    standard counterparts.
    """
    
    def __init__(self, name: str):
        """
        Initialize degraded mode indicator.
        
        Args:
            name: Name of the indicator
        """
        self.name = name
        self.logger = logging.getLogger(f"DegradedIndicator.{name}")
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate simplified indicator values.
        
        Args:
            data: Input data for calculation
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with indicator values
        """
        pass
        
    def estimate_complexity(self, data_size: int) -> float:
        """
        Estimate computational complexity of the indicator.
        
        Args:
            data_size: Size of input data
            
        Returns:
            Estimated complexity score (higher means more complex)
        """
        # Default implementation - should be overridden by complex indicators
        # Simple indicators typically have linear complexity
        return data_size
        
    def compare_to_standard(
        self, 
        standard_result: pd.DataFrame, 
        degraded_result: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compare degraded mode results to standard results.
        
        Args:
            standard_result: Result from standard calculation
            degraded_result: Result from degraded mode calculation
            
        Returns:
            Dictionary with comparison metrics
        """
        # Find common columns between the results
        common_columns = [col for col in standard_result.columns 
                          if col in degraded_result.columns]
        
        metrics = {}
        
        for col in common_columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(standard_result[col]) or \
               not pd.api.types.is_numeric_dtype(degraded_result[col]):
                continue
                
            # Get values, dropping NaN
            std_values = standard_result[col].dropna()
            deg_values = degraded_result[col].dropna()
            
            # Use common index
            common_idx = std_values.index.intersection(deg_values.index)
            if len(common_idx) < 2:
                continue
                
            std_values = std_values.loc[common_idx]
            deg_values = deg_values.loc[common_idx]
            
            # Calculate error metrics
            metrics[f"{col}_mae"] = np.abs(std_values - deg_values).mean()
            metrics[f"{col}_rmse"] = np.sqrt(np.mean((std_values - deg_values) ** 2))
            metrics[f"{col}_max_error"] = np.abs(std_values - deg_values).max()
            
            # Calculate correlation
            try:
                metrics[f"{col}_correlation"] = std_values.corr(deg_values)
            except:
                metrics[f"{col}_correlation"] = np.nan
        
        return metrics


def degraded_indicator(
    standard_indicator_class: Optional[Type[BaseIndicator]] = None,
    estimated_speedup: float = 2.0,
    accuracy_loss: float = 0.1,  # 0.0=perfect, 1.0=useless
    activation_threshold: float = 0.6  # System load threshold (0-1)
):
    """
    Decorator for degraded mode indicator implementation.
    
    This decorator registers a degraded mode indicator class and connects
    it with its standard implementation for automatic switching.
    
    Args:
        standard_indicator_class: The standard indicator class this degrades
        estimated_speedup: Estimated performance improvement factor
        accuracy_loss: Estimated accuracy loss compared to standard
        activation_threshold: System load threshold to activate degraded mode
        
    Returns:
        Decorated class
    """
    def decorator(degraded_class):
        # Register metadata about the degraded indicator
        degraded_class._degraded_mode_metadata = {
            'standard_class': standard_indicator_class.__name__ if standard_indicator_class else None,
            'estimated_speedup': estimated_speedup,
            'accuracy_loss': accuracy_loss,
            'activation_threshold': activation_threshold
        }
        
        # Add a method to check if degraded mode should be used
        def should_use_degraded(self, current_load: float = 0.0) -> bool:
            # Use degraded mode if the load exceeds the threshold
            # or if explicitly requested
            return current_load >= self._degraded_mode_metadata['activation_threshold']
            
        degraded_class.should_use_degraded = should_use_degraded
        
        # Add metadata accessor
        def get_degraded_metadata(self):
            return self._degraded_mode_metadata
            
        degraded_class.get_degraded_metadata = get_degraded_metadata
        
        return degraded_class
        
    # Handle case where decorator is used without arguments
    if standard_indicator_class is not None and callable(standard_indicator_class):
        cls = standard_indicator_class
        standard_indicator_class = None
        return decorator(cls)
        
    return decorator


class AdaptiveComplexityIndicator(BaseIndicator):
    """
    Indicator that can adapt its complexity based on system load.
    
    This base class provides functionality to switch between standard and
    degraded mode calculations based on system conditions.
    """
    
    def __init__(self, name: str, degraded_implementation: Optional[DegradedModeIndicator] = None):
        """
        Initialize with both standard and degraded implementations.
        
        Args:
            name: Indicator name
            degraded_implementation: Degraded mode implementation
        """
        super().__init__(name)
        self._degraded_implementation = degraded_implementation
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate indicator using appropriate implementation based on load.
        
        Args:
            data: Input data
            **kwargs: Additional parameters including degradation_level
            
        Returns:
            DataFrame with indicator results
        """
        # Check for degradation_level parameter
        degradation_level = kwargs.pop('degradation_level', 0.0)
        
        # Decide which implementation to use
        if self._degraded_implementation and degradation_level > 0.5:
            self.logger.debug(f"Using degraded implementation for {self.name}")
            result = self._degraded_implementation.calculate(data, **kwargs)
            # Mark result as degraded
            if hasattr(result, 'attrs'):
                result.attrs['degraded'] = True
                result.attrs['degradation_level'] = degradation_level
            return result
        else:
            # Use standard implementation
            return self._calculate_standard(data, **kwargs)
            
    def _calculate_standard(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Standard calculation implementation.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with indicator results
        """
        # Default implementation delegates to super(), should be overridden
        # by subclasses that inherit from specific indicator types
        return super().calculate(data, **kwargs)
        
    def calculate_degraded(self, data: pd.DataFrame, degradation_level: float = 1.0, **kwargs) -> pd.DataFrame:
        """
        Explicitly use degraded calculation regardless of system load.
        
        Args:
            data: Input data
            degradation_level: Level of degradation (0-1)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with indicator results
        """
        if self._degraded_implementation:
            result = self._degraded_implementation.calculate(data, **kwargs)
            # Mark result as explicitly degraded
            if hasattr(result, 'attrs'):
                result.attrs['degraded'] = True
                result.attrs['degradation_level'] = degradation_level
            return result
        else:
            self.logger.warning(f"No degraded implementation available for {self.name}, using standard")
            return self._calculate_standard(data, **kwargs)
            
    def set_degraded_implementation(self, implementation: DegradedModeIndicator) -> None:
        """
        Set degraded mode implementation.
        
        Args:
            implementation: Degraded mode indicator implementation
        """
        self._degraded_implementation = implementation
