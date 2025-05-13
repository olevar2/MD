"""
Moving Average Incremental Indicators

This module provides incremental implementations of various moving average indicators
optimized for memory usage and processing efficiency.
"""
from typing import Dict, List, Optional, Any
import numpy as np
from core.base_6 import (
    ValueWindowIndicator,
    RecursiveIndicator,
    CompositeIndicator,
    IncrementalIndicator
)


class SMA(ValueWindowIndicator):
    """
    Simple Moving Average - Incremental Implementation
    
    Calculates average of prices over a specified window period.
    This implementation uses a sliding window approach that's
    memory efficient and O(1) complexity per update.
    """
    
    def __init__(self, window_size: int, price_key: str = 'close'):
        """
        Initialize Simple Moving Average
        
        Args:
            window_size: Period for the moving average
            price_key: Key to extract prices from data points
        """
        super().__init__(f"SMA_{window_size}", window_size, price_key)
        self.sum = 0.0
    
    def _calculate(self) -> Dict[str, Any]:
        """Calculate SMA from current window"""
        if len(self.values) < self.window_size:
            # Not enough data for a complete calculation
            return {'value': None, 'complete': False}
        
        # Update running sum if not already tracking
        if self.sum == 0.0 and len(self.values) > 0:
            self.sum = sum(self.values)
            
        self.sum = sum(self.values)  # Recalculate for safety
        sma_value = self.sum / len(self.values)
        
        return {'value': sma_value, 'complete': True}
    
    def update(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimized update for SMA that maintains a running sum
        
        Args:
            data_point: New price data
            
        Returns:
            Updated SMA value
        """
        if not self.is_initialized:
            return {'value': None, 'complete': False}
        
        if self.price_key not in data_point:
            return {'value': None, 'complete': False}
        
        price = data_point[self.price_key]
        
        # Update running sum and window
        self.sum += price
        self.values.append(price)
        
        # Remove oldest value if window is full
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        
        # Calculate SMA
        if len(self.values) == self.window_size:
            return {'value': self.sum / self.window_size, 'complete': True}
        else:
            return {'value': None, 'complete': False}
    
    def reset(self) -> None:
        """Reset indicator state"""
        super().reset()
        self.sum = 0.0
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for persistence"""
        state = super().get_state()
        state['sum'] = self.sum
        return state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore from saved state"""
        super().set_state(state)
        if 'sum' in state:
            self.sum = state['sum']
        else:
            # Recalculate sum if not provided
            self.sum = sum(self.values) if self.values else 0.0


class EMA(RecursiveIndicator):
    """
    Exponential Moving Average - Incremental Implementation
    
    EMA gives more weight to recent prices compared to SMA.
    This implementation uses the recursive formula for EMA
    which is more efficient than recalculating from a window.
    """
    
    def __init__(
        self,
        window_size: int,
        price_key: str = 'close',
        smoothing: float = 2.0
    ):
        """
        Initialize Exponential Moving Average
        
        Args:
            window_size: Period for the moving average
            price_key: Key to extract prices from data points
            smoothing: Smoothing factor (typically 2.0)
        """
        params = {'smoothing': smoothing}
        super().__init__(f"EMA_{window_size}", window_size, price_key, params)
        self.smoothing = smoothing
        self.alpha = smoothing / (window_size + 1)
        self.state['last_ema'] = None
    
    def _calculate_recursive(self, new_price: float) -> Dict[str, Any]:
        """
        Calculate new EMA value based on previous EMA and new price
        
        Args:
            new_price: New price value
            
        Returns:
            Dictionary with EMA value
        """
        last_ema = self.state.get('last_ema')
        
        # First value initialization
        if last_ema is None:
            if self.state['values_seen'] >= self.window_size:
                # Typically first EMA = first price, but here we're assuming
                # we've seen enough values to start the EMA calculation
                self.state['last_ema'] = new_price
                return {'value': new_price, 'complete': True}
            else:
                return {'value': None, 'complete': False}
        
        # Regular EMA calculation: EMA = Price * α + Previous EMA * (1 - α)
        new_ema = new_price * self.alpha + last_ema * (1 - self.alpha)
        self.state['last_ema'] = new_ema
        
        return {'value': new_ema, 'complete': True}
    
    def _initialize_with_prices(self, prices: List[float]) -> None:
        """
        Initialize EMA with historical prices
        
        Args:
            prices: List of historical prices
        """
        if not prices:
            return
        
        # Record how many values we've seen
        self.state['values_seen'] = len(prices)
        
        # Need at least window_size values for initialization
        if len(prices) < self.window_size:
            return
        
        # Calculate initial SMA
        initial_sma = sum(prices[:self.window_size]) / self.window_size
        
        # Use SMA as first EMA
        ema = initial_sma
        
        # Calculate EMA for remaining prices
        for price in prices[self.window_size:]:
            ema = price * self.alpha + ema * (1 - self.alpha)
        
        self.state['last_ema'] = ema


class WMA(ValueWindowIndicator):
    """
    Weighted Moving Average - Incremental Implementation
    
    WMA gives more weight to recent prices, with weights decreasing linearly.
    """
    
    def __init__(self, window_size: int, price_key: str = 'close'):
        """
        Initialize Weighted Moving Average
        
        Args:
            window_size: Period for the moving average
            price_key: Key to extract prices from data points
        """
        super().__init__(f"WMA_{window_size}", window_size, price_key)
        
        # Pre-calculate weights for efficiency
        self.weights = np.arange(1, window_size + 1)
        self.weight_sum = np.sum(self.weights)
    
    def _calculate(self) -> Dict[str, Any]:
        """Calculate WMA from current window"""
        if len(self.values) < self.window_size:
            return {'value': None, 'complete': False}
        
        # Apply weights (higher weights for more recent values)
        values_array = np.array(self.values)
        weighted_sum = np.sum(values_array * self.weights) / self.weight_sum
        
        return {'value': weighted_sum, 'complete': True}


class MACD(CompositeIndicator):
    """
    Moving Average Convergence/Divergence - Incremental Implementation
    
    MACD is calculated using two EMAs (typically 12 and 26 periods)
    and includes a signal line (typically 9-period EMA of MACD).
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        price_key: str = 'close'
    ):
        """
        Initialize MACD indicator
        
        Args:
            fast_period: Period for the fast EMA
            slow_period: Period for the slow EMA
            signal_period: Period for the signal line EMA
            price_key: Key to extract prices from data points
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.price_key = price_key
        
        # Create sub-indicators
        self.fast_ema = EMA(fast_period, price_key)
        self.slow_ema = EMA(slow_period, price_key)
        
        # Composite indicator requires list of sub-indicators
        sub_indicators = [self.fast_ema, self.slow_ema]
        
        params = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'price_key': price_key
        }
        
        super().__init__(
            f"MACD_{fast_period}_{slow_period}_{signal_period}",
            sub_indicators,
            params
        )
        
        # MACD line values for signal line calculation
        self.macd_values = []
        self.signal_ema = None
        self.state['last_macd'] = None
        self.state['last_signal'] = None
        self.state['macd_count'] = 0
    
    def _combine_results(self, sub_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine EMA results to calculate MACD
        
        Args:
            sub_results: Results from fast and slow EMAs
            
        Returns:
            Dictionary with MACD values
        """
        fast_ema_result = sub_results.get(f"EMA_{self.fast_period}", {}).get('value')
        slow_ema_result = sub_results.get(f"EMA_{self.slow_period}", {}).get('value')
        
        # Need both EMAs to calculate MACD
        if fast_ema_result is None or slow_ema_result is None:
            return {
                'macd': None,
                'signal': None,
                'histogram': None,
                'complete': False
            }
        
        # Calculate MACD line (fast EMA - slow EMA)
        macd_value = fast_ema_result - slow_ema_result
        self.state['last_macd'] = macd_value
        
        # Add to MACD values for signal line
        self.macd_values.append(macd_value)
        self.state['macd_count'] += 1
        
        # Keep only signal_period most recent values
        if len(self.macd_values) > self.signal_period:
            self.macd_values.pop(0)
        
        # Calculate signal line (EMA of MACD)
        signal_value = None
        if len(self.macd_values) == self.signal_period:
            # Initialize signal on first complete window
            if self.signal_ema is None:
                self.signal_ema = sum(self.macd_values) / self.signal_period
            else:
                # Update signal using EMA formula
                alpha = 2.0 / (self.signal_period + 1)
                self.signal_ema = macd_value * alpha + self.signal_ema * (1 - alpha)
            
            signal_value = self.signal_ema
        
        self.state['last_signal'] = signal_value
        
        # Calculate histogram
        histogram_value = None
        if macd_value is not None and signal_value is not None:
            histogram_value = macd_value - signal_value
        
        return {
            'macd': macd_value,
            'signal': signal_value,
            'histogram': histogram_value,
            'complete': signal_value is not None
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state including MACD values"""
        state = super().get_state()
        state['macd_values'] = self.macd_values.copy()
        state['signal_ema'] = self.signal_ema
        return state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state including MACD values"""
        super().set_state(state)
        if 'macd_values' in state:
            self.macd_values = state['macd_values']
        if 'signal_ema' in state:
            self.signal_ema = state['signal_ema']
    
    def reset(self) -> None:
        """Reset indicator state"""
        super().reset()
        self.macd_values = []
        self.signal_ema = None
