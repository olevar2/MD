"""
Incremental Indicator Base Classes

This module provides the foundation for incremental technical indicator calculation
with optimized memory and processing efficiency for low-latency environments.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import logging
from datetime import datetime


class IncrementalIndicator(ABC):
    """
    Abstract base class for all incremental technical indicators
    
    This class defines the interface for indicators that can be updated
    incrementally with each new data point, without recalculating the 
    entire history, optimizing for memory usage and processing time.
    """
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        """
        Initialize incremental indicator
        
        Args:
            name: Name of the indicator
            params: Optional parameters for the indicator
        """
        self.name = name
        self.params = params or {}
        self.is_initialized = False
        self.last_update_time = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def update(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update indicator state with a new data point
        
        Args:
            data_point: New data point containing required values (e.g. OHLCV)
            
        Returns:
            Dictionary containing updated indicator value(s)
        """
        pass
    
    @abstractmethod
    def initialize(self, historical_data: List[Dict[str, Any]]) -> None:
        """
        Initialize indicator state with historical data
        
        Args:
            historical_data: List of historical data points
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the indicator for persistence"""
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore the state of the indicator
        
        Args:
            state: Previously saved state
        """
        pass
    
    def reset(self) -> None:
        """Reset indicator state"""
        self.is_initialized = False
        self.last_update_time = None


class ValueWindowIndicator(IncrementalIndicator):
    """
    Base class for indicators that require a sliding window of values
    
    This class implements efficient windowing for indicators like SMA
    that need to track a specific number of previous values.
    """
    
    def __init__(
        self, 
        name: str, 
        window_size: int, 
        price_key: str = 'close',
        params: Dict[str, Any] = None
    ):
        """
        Initialize with window size and price key
        
        Args:
            name: Name of the indicator
            window_size: Size of the window to maintain
            price_key: Key to extract price values from data points
            params: Additional parameters
        """
        super().__init__(name, params or {})
        self.window_size = window_size
        self.price_key = price_key
        self.values = []  # Circular buffer would be more efficient for large windows
        self.params.update({
            'window_size': window_size,
            'price_key': price_key
        })
    
    def update(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update with a new data point
        
        Args:
            data_point: New data point with required price
            
        Returns:
            Dictionary with indicator value
        """
        if not self.is_initialized:
            self.logger.warning(f"{self.name} must be initialized before updating")
            return {'value': None}
        
        if self.price_key not in data_point:
            self.logger.warning(f"Price key '{self.price_key}' not found in data point")
            return {'value': None}
        
        # Extract price and update timestamp
        price = data_point[self.price_key]
        
        # Update timestamp
        if 'timestamp' in data_point:
            self.last_update_time = data_point['timestamp']
        else:
            self.last_update_time = datetime.now()
        
        # Add new value to sliding window
        self.values.append(price)
        
        # Remove oldest value if window is full
        if len(self.values) > self.window_size:
            self.values.pop(0)
        
        # Calculate the indicator value
        return self._calculate()
    
    def initialize(self, historical_data: List[Dict[str, Any]]) -> None:
        """
        Initialize with historical data
        
        Args:
            historical_data: List of historical data points
        """
        self.reset()
        
        if not historical_data:
            self.logger.warning("No historical data provided for initialization")
            return
        
        # Extract price values from historical data
        for data_point in historical_data[-self.window_size:]:
            if self.price_key in data_point:
                self.values.append(data_point[self.price_key])
        
        if historical_data and 'timestamp' in historical_data[-1]:
            self.last_update_time = historical_data[-1]['timestamp']
        
        self.is_initialized = True
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for persistence"""
        return {
            'name': self.name,
            'params': self.params,
            'values': self.values.copy(),
            'is_initialized': self.is_initialized,
            'last_update_time': self.last_update_time
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore from saved state"""
        if 'values' in state:
            self.values = state['values']
        if 'is_initialized' in state:
            self.is_initialized = state['is_initialized']
        if 'last_update_time' in state:
            self.last_update_time = state['last_update_time']
    
    def reset(self) -> None:
        """Reset state"""
        super().reset()
        self.values = []
    
    @abstractmethod
    def _calculate(self) -> Dict[str, Any]:
        """
        Calculate indicator value from current window
        
        Returns:
            Dictionary with indicator value(s)
        """
        pass


class StatefulIndicator(IncrementalIndicator):
    """
    Base class for indicators that maintain internal state beyond a simple window
    
    This class is for indicators like EMA, RSI, or MACD that require
    more complex state tracking beyond simple value windows.
    """
    
    def __init__(
        self, 
        name: str, 
        params: Dict[str, Any] = None
    ):
        """
        Initialize stateful indicator
        
        Args:
            name: Name of the indicator
            params: Configuration parameters
        """
        super().__init__(name, params or {})
        self.state = {}  # Internal state storage
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for persistence"""
        return {
            'name': self.name,
            'params': self.params,
            'state': self.state.copy(),
            'is_initialized': self.is_initialized,
            'last_update_time': self.last_update_time
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore from saved state"""
        if 'state' in state:
            self.state = state['state']
        if 'is_initialized' in state:
            self.is_initialized = state['is_initialized']
        if 'last_update_time' in state:
            self.last_update_time = state['last_update_time']
    
    def reset(self) -> None:
        """Reset state"""
        super().reset()
        self.state = {}


class RecursiveIndicator(StatefulIndicator):
    """
    Base class for indicators that calculate new values based on previous results
    
    This class is for indicators like EMA that use their previous output
    to calculate the next value, creating a recursive calculation pattern.
    """
    
    def __init__(
        self, 
        name: str, 
        window_size: int,
        price_key: str = 'close',
        params: Dict[str, Any] = None
    ):
        """
        Initialize recursive indicator
        
        Args:
            name: Name of the indicator
            window_size: Lookback period for the indicator
            price_key: Key to extract price values from data points
            params: Additional parameters
        """
        params = params or {}
        params.update({
            'window_size': window_size,
            'price_key': price_key
        })
        super().__init__(name, params)
        self.window_size = window_size
        self.price_key = price_key
        self.last_value = None
        self.state['values_seen'] = 0
    
    def update(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update with a new data point
        
        Args:
            data_point: New data point with required price
            
        Returns:
            Dictionary with indicator value
        """
        if not self.is_initialized and 'values_seen' not in self.state:
            self.logger.warning(f"{self.name} must be initialized before updating")
            return {'value': None}
        
        if self.price_key not in data_point:
            self.logger.warning(f"Price key '{self.price_key}' not found in data point")
            return {'value': None}
        
        # Update timestamp
        if 'timestamp' in data_point:
            self.last_update_time = data_point['timestamp']
        else:
            self.last_update_time = datetime.now()
        
        # Update value count
        self.state['values_seen'] += 1
        
        # Calculate the new indicator value
        return self._calculate_recursive(data_point[self.price_key])
    
    @abstractmethod
    def _calculate_recursive(self, new_price: float) -> Dict[str, Any]:
        """
        Calculate new indicator value based on the previous value and new price
        
        Args:
            new_price: New price value
            
        Returns:
            Dictionary with indicator value(s)
        """
        pass
    
    def initialize(self, historical_data: List[Dict[str, Any]]) -> None:
        """
        Initialize with historical data
        
        Args:
            historical_data: List of historical data points
        """
        self.reset()
        
        if not historical_data:
            self.logger.warning("No historical data provided for initialization")
            return
        
        # For recursive indicators, we often need to calculate the entire series
        # to establish the correct current value
        
        # Get relevant historical prices
        prices = []
        for data_point in historical_data:
            if self.price_key in data_point:
                prices.append(data_point[self.price_key])
        
        # If we don't have enough data, initialize with what we have
        if len(prices) < self.window_size:
            self.logger.warning(
                f"Insufficient historical data for {self.name}. "
                f"Need {self.window_size}, got {len(prices)}"
            )
        
        # Initialize with available data
        self._initialize_with_prices(prices)
        
        if historical_data and 'timestamp' in historical_data[-1]:
            self.last_update_time = historical_data[-1]['timestamp']
        
        self.is_initialized = True
    
    @abstractmethod
    def _initialize_with_prices(self, prices: List[float]) -> None:
        """
        Initialize internal state using historical price data
        
        Args:
            prices: List of historical prices
        """
        pass


class CompositeIndicator(StatefulIndicator):
    """
    Indicator composed of multiple sub-indicators
    
    This class allows building complex indicators from simpler ones,
    like MACD which uses multiple EMAs.
    """
    
    def __init__(
        self, 
        name: str,
        sub_indicators: List[IncrementalIndicator],
        params: Dict[str, Any] = None
    ):
        """
        Initialize with sub-indicators
        
        Args:
            name: Name of the composite indicator
            sub_indicators: List of component indicators
            params: Additional parameters
        """
        super().__init__(name, params or {})
        self.sub_indicators = sub_indicators
    
    def update(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update all sub-indicators and combine their results
        
        Args:
            data_point: New data point
            
        Returns:
            Dictionary with combined indicator values
        """
        if not self.is_initialized:
            self.logger.warning(f"{self.name} must be initialized before updating")
            return {'value': None}
        
        # Update timestamp
        if 'timestamp' in data_point:
            self.last_update_time = data_point['timestamp']
        else:
            self.last_update_time = datetime.now()
        
        # Update all sub-indicators
        sub_results = {}
        for indicator in self.sub_indicators:
            result = indicator.update(data_point)
            sub_results[indicator.name] = result
        
        # Combine results
        return self._combine_results(sub_results)
    
    def initialize(self, historical_data: List[Dict[str, Any]]) -> None:
        """
        Initialize all sub-indicators
        
        Args:
            historical_data: List of historical data points
        """
        self.reset()
        
        if not historical_data:
            self.logger.warning("No historical data provided for initialization")
            return
        
        # Initialize all sub-indicators
        for indicator in self.sub_indicators:
            indicator.initialize(historical_data)
        
        if historical_data and 'timestamp' in historical_data[-1]:
            self.last_update_time = historical_data[-1]['timestamp']
        
        self.is_initialized = True
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state including all sub-indicators"""
        sub_states = {}
        for indicator in self.sub_indicators:
            sub_states[indicator.name] = indicator.get_state()
        
        state = super().get_state()
        state['sub_states'] = sub_states
        return state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state including all sub-indicators"""
        super().set_state(state)
        
        if 'sub_states' in state:
            sub_states = state['sub_states']
            for indicator in self.sub_indicators:
                if indicator.name in sub_states:
                    indicator.set_state(sub_states[indicator.name])
    
    def reset(self) -> None:
        """Reset composite indicator and all sub-indicators"""
        super().reset()
        for indicator in self.sub_indicators:
            indicator.reset()
    
    @abstractmethod
    def _combine_results(self, sub_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine results from sub-indicators
        
        Args:
            sub_results: Dictionary of results from sub-indicators
            
        Returns:
            Combined indicator values
        """
        pass
