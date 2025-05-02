"""
Base Incremental Indicator Module.

This module provides the foundation for implementing incremental technical indicators
that can efficiently update their values when new data arrives without recalculating
the entire dataset, which is critical for low-latency applications.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from core_foundations.utils.logger import get_logger

logger = get_logger("feature-store-service.incremental-indicators")


class IncrementalIndicator(ABC):
    """
    Base class for incremental technical indicators.
    
    This abstract class defines the interface for indicators that can be
    computed incrementally, maintaining state between updates to avoid
    full recalculation when new data arrives.
    """
    
    def __init__(self, name: str, params: Dict[str, Any]):
        """
        Initialize the incremental indicator.
        
        Args:
            name: Name of the indicator
            params: Dictionary of parameters for the indicator
        """
        self.name = name
        self.params = params
        self.is_initialized = False
        self.state = {}  # Internal state for incremental calculation
        self.last_timestamp = None
        
    @abstractmethod
    def initialize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Initialize the indicator with historical data.
        
        This method computes the initial values of the indicator on historical data
        and sets up the internal state for future incremental updates.
        
        Args:
            data: Historical OHLCV data
            
        Returns:
            DataFrame with the indicator values added
        """
        pass
    
    @abstractmethod
    def update(self, new_data_point: Dict[str, Union[float, datetime]]) -> Dict[str, float]:
        """
        Update the indicator with a new data point.
        
        This method incrementally updates the indicator with a new data point
        without recalculating from the beginning of the time series.
        
        Args:
            new_data_point: Dictionary containing a new OHLCV data point
                            (keys: open, high, low, close, volume, timestamp)
            
        Returns:
            Dictionary containing the updated indicator values
        """
        pass
    
    def get_output_columns(self) -> List[str]:
        """
        Get the names of the output columns produced by this indicator.
        
        Returns:
            List of column names
        """
        return [f"{self.name}"]
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current internal state of the indicator.
        
        This method can be used to serialize the indicator state for persistence.
        
        Returns:
            Dictionary containing the internal state
        """
        return {
            "name": self.name,
            "params": self.params,
            "is_initialized": self.is_initialized,
            "state": self.state,
            "last_timestamp": self.last_timestamp
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the internal state of the indicator.
        
        This method can be used to deserialize a previously saved state.
        
        Args:
            state: Dictionary containing the internal state
        """
        self.name = state.get("name", self.name)
        self.params = state.get("params", self.params)
        self.is_initialized = state.get("is_initialized", False)
        self.state = state.get("state", {})
        self.last_timestamp = state.get("last_timestamp", None)
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the indicator to a DataFrame of OHLCV data.
        
        This method is a convenience wrapper that initializes the indicator
        if it's not already initialized, or calls update for each new row
        if the indicator is already initialized.
        
        Args:
            data: OHLCV data
            
        Returns:
            DataFrame with the indicator values added
        """
        if not self.is_initialized:
            return self.initialize(data)
        else:
            # For each new row that was not included in the last initialization,
            # call the update method incrementally
            result = data.copy()
            
            # Find rows after the last timestamp
            if self.last_timestamp is not None:
                new_rows = result.loc[result.index > self.last_timestamp].copy()
                
                # For each new row, call update
                for idx, row in new_rows.iterrows():
                    data_point = row.to_dict()
                    data_point['timestamp'] = idx
                    
                    # Update the indicator
                    updated_values = self.update(data_point)
                    
                    # Add the updated values to the result DataFrame
                    for col_name, value in updated_values.items():
                        result.at[idx, col_name] = value
            
            return result


class IncrementalIndicatorFactory:
    """
    Factory for creating incremental indicator instances.
    
    This class is responsible for creating instances of incremental
    indicators based on their name and parameters.
    """
    
    @classmethod
    def create_indicator(cls, indicator_type: str, **params) -> Optional[IncrementalIndicator]:
        """
        Create an incremental indicator instance.
        
        Args:
            indicator_type: Type of indicator to create (e.g., "SMA", "EMA")
            **params: Parameters for the indicator
            
        Returns:
            IncrementalIndicator instance or None if the type is not supported
        """
        from feature_store_service.computation.incremental.moving_averages import (
            IncrementalSMA, IncrementalEMA
        )
        
        indicator_map = {
            "SMA": IncrementalSMA,
            "EMA": IncrementalEMA,
            # Add more indicators as they are implemented
        }
        
        indicator_class = indicator_map.get(indicator_type)
        if not indicator_class:
            logger.error(f"Unsupported incremental indicator type: {indicator_type}")
            return None
            
        return indicator_class(name=indicator_type, params=params)