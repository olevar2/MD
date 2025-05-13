"""
Volatility Incremental Indicators

This module provides incremental implementations of volatility indicators
optimized for memory usage and processing efficiency.
"""
from typing import Dict, List, Optional, Any
import numpy as np
import math
from core.base_6 import (
    StatefulIndicator,
    ValueWindowIndicator,
    RecursiveIndicator
)


class ATR(RecursiveIndicator):
    """
    Average True Range - Incremental Implementation
    
    ATR measures market volatility by decomposing the range of an asset price.
    This implementation uses Wilder's smoothing method.
    """
    
    def __init__(self, window_size: int = 14):
        """
        Initialize ATR indicator
        
        Args:
            window_size: Lookback period for ATR calculation (default 14)
        """
        super().__init__(f"ATR_{window_size}", window_size, 'close')
        self.state = {
            'values_seen': 0,
            'tr_values': [],
            'last_atr': None,
            'prev_high': None,
            'prev_low': None,
            'prev_close': None
        }
    
    def _calculate_recursive(self, new_price: float) -> Dict[str, Any]:
        """
        Calculate new ATR value recursively
        
        Args:
            new_price: New close price
            
        Returns:
            Dictionary with ATR value
        """
        # ATR requires high, low, close values, not just a price
        # This method will be called from update() which has already
        # extracted the whole data point
        return {'value': self.state['last_atr']}
    
    def update(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update ATR with a new data point
        
        Args:
            data_point: New data point with high, low, close prices
            
        Returns:
            Dictionary with ATR value
        """
        if not self.is_initialized:
            return {'value': None}
        
        # ATR needs high, low, and close prices
        if 'high' not in data_point or 'low' not in data_point or 'close' not in data_point:
            return {'value': None}
        
        # Update timestamp
        if 'timestamp' in data_point:
            self.last_update_time = data_point['timestamp']
        
        high = data_point['high']
        low = data_point['low']
        close = data_point['close']
        
        prev_high = self.state.get('prev_high')
        prev_low = self.state.get('prev_low')
        prev_close = self.state.get('prev_close')
        
        # First data point, just store values
        if prev_close is None:
            self.state['prev_high'] = high
            self.state['prev_low'] = low
            self.state['prev_close'] = close
            self.state['values_seen'] += 1
            return {'value': None}
        
        # Calculate True Range
        tr = max(
            high - low,  # Current high - low
            abs(high - prev_close),  # Current high - previous close
            abs(low - prev_close)  # Current low - previous close
        )
        
        # Update state
        self.state['values_seen'] += 1
        
        # Initial collection period
        if self.state['values_seen'] <= self.window_size:
            self.state['tr_values'].append(tr)
            
            # After window_size points, calculate initial ATR
            if self.state['values_seen'] == self.window_size:
                self.state['last_atr'] = sum(self.state['tr_values']) / self.window_size
            
            # Update previous values and return
            self.state['prev_high'] = high
            self.state['prev_low'] = low
            self.state['prev_close'] = close
            
            return {'value': self.state['last_atr']}
        
        # Subsequent calculations use Wilder's smoothing
        last_atr = self.state['last_atr']
        alpha = 1 / self.window_size
        
        # ATR = [(Prior ATR × (n-1)) + Current TR] / n
        new_atr = ((last_atr * (self.window_size - 1)) + tr) / self.window_size
        self.state['last_atr'] = new_atr
        
        # Update previous values for next calculation
        self.state['prev_high'] = high
        self.state['prev_low'] = low
        self.state['prev_close'] = close
        
        return {'value': new_atr}
    
    def _initialize_with_prices(self, prices: List[float]) -> None:
        """
        ATR requires high, low, close prices, not just a single price series
        This method is not directly used for ATR initialization
        """
        pass
    
    def initialize(self, historical_data: List[Dict[str, Any]]) -> None:
        """
        Initialize ATR with historical data
        
        Args:
            historical_data: List of historical data points with high, low, close
        """
        self.reset()
        
        if not historical_data:
            return
        
        # Need at least window_size + 1 data points for a proper ATR calculation
        if len(historical_data) <= self.window_size:
            self.logger.warning(
                f"Insufficient historical data for {self.name}. "
                f"Need at least {self.window_size + 1} points."
            )
            return
        
        # Process historical data to calculate initial ATR
        prev_close = None
        tr_values = []
        
        for i, data_point in enumerate(historical_data):
            if not all(k in data_point for k in ['high', 'low', 'close']):
                continue
            
            high = data_point['high']
            low = data_point['low']
            close = data_point['close']
            
            if prev_close is not None:
                # Calculate True Range
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                tr_values.append(tr)
            
            prev_close = close
            
            # Store last data point values
            if i == len(historical_data) - 1:
                self.state['prev_high'] = high
                self.state['prev_low'] = low
                self.state['prev_close'] = close
        
        # Keep only the most recent window_size TR values
        tr_values = tr_values[-self.window_size:]
        
        if len(tr_values) < self.window_size:
            self.logger.warning(
                f"Not enough valid data points for {self.name} calculation"
            )
            return
        
        # Calculate initial ATR as simple average of TR values
        initial_atr = sum(tr_values) / self.window_size
        
        # Update state
        self.state['last_atr'] = initial_atr
        self.state['values_seen'] = self.window_size
        self.state['tr_values'] = tr_values
        
        if historical_data and 'timestamp' in historical_data[-1]:
            self.last_update_time = historical_data[-1]['timestamp']
        
        self.is_initialized = True


class BollingerBands(StatefulIndicator):
    """
    Bollinger Bands - Incremental Implementation
    
    Bollinger Bands consist of a middle band (typically SMA) and two outer bands
    that are standard deviations away from the middle band.
    """
    
    def __init__(
        self,
        window_size: int = 20,
        num_std_dev: float = 2.0,
        price_key: str = 'close'
    ):
        """
        Initialize Bollinger Bands indicator
        
        Args:
            window_size: Window period for the middle band calculation (default 20)
            num_std_dev: Number of standard deviations for the bands (default 2)
            price_key: Key to extract prices from data points
        """
        params = {
            'window_size': window_size,
            'num_std_dev': num_std_dev,
            'price_key': price_key
        }
        super().__init__(f"BollingerBands_{window_size}_{num_std_dev}", params)
        
        self.window_size = window_size
        self.num_std_dev = num_std_dev
        self.price_key = price_key
        
        # Initialize state
        self.state = {
            'prices': [],
            'sum': 0.0,
            'sum_squares': 0.0
        }
    
    def update(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update Bollinger Bands with a new data point
        
        Args:
            data_point: New data point
            
        Returns:
            Dictionary with middle, upper, and lower band values
        """
        if not self.is_initialized:
            return {'middle': None, 'upper': None, 'lower': None}
        
        if self.price_key not in data_point:
            return {'middle': None, 'upper': None, 'lower': None}
        
        # Update timestamp
        if 'timestamp' in data_point:
            self.last_update_time = data_point['timestamp']
        
        price = data_point[self.price_key]
        
        # Update running sums for efficiency
        self.state['sum'] += price
        self.state['sum_squares'] += price * price
        self.state['prices'].append(price)
        
        # Remove oldest price if window is full
        if len(self.state['prices']) > self.window_size:
            oldest_price = self.state['prices'].pop(0)
            self.state['sum'] -= oldest_price
            self.state['sum_squares'] -= oldest_price * oldest_price
        
        # Calculate bands once we have enough data
        if len(self.state['prices']) == self.window_size:
            # Calculate middle band (SMA)
            middle_band = self.state['sum'] / self.window_size
            
            # Calculate standard deviation
            variance = (self.state['sum_squares'] / self.window_size) - (middle_band * middle_band)
            std_dev = math.sqrt(max(0, variance))  # Prevent negative variance
            
            # Calculate upper and lower bands
            upper_band = middle_band + (self.num_std_dev * std_dev)
            lower_band = middle_band - (self.num_std_dev * std_dev)
            
            return {
                'middle': middle_band,
                'upper': upper_band,
                'lower': lower_band,
                'bandwidth': (upper_band - lower_band) / middle_band if middle_band != 0 else None
            }
        
        return {'middle': None, 'upper': None, 'lower': None}
    
    def initialize(self, historical_data: List[Dict[str, Any]]) -> None:
        """
        Initialize Bollinger Bands with historical data
        
        Args:
            historical_data: List of historical data points
        """
        self.reset()
        
        if not historical_data:
            return
        
        # Extract prices from historical data
        prices = []
        for data_point in historical_data:
            if self.price_key in data_point:
                prices.append(data_point[self.price_key])
        
        # Use only the window_size most recent prices
        prices = prices[-self.window_size:]
        
        if len(prices) < self.window_size:
            self.logger.warning(
                f"Insufficient historical data for {self.name}. "
                f"Need at least {self.window_size} points."
            )
            return
        
        # Store prices and calculate sums
        self.state['prices'] = prices
        self.state['sum'] = sum(prices)
        self.state['sum_squares'] = sum(p*p for p in prices)
        
        if historical_data and 'timestamp' in historical_data[-1]:
            self.last_update_time = historical_data[-1]['timestamp']
        
        self.is_initialized = True


class Keltner(StatefulIndicator):
    """
    Keltner Channels - Incremental Implementation
    
    Keltner Channels use ATR for band width instead of standard deviation
    as in Bollinger Bands.
    """
    
    def __init__(
        self,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
        price_key: str = 'close'
    ):
        """
        Initialize Keltner Channels
        
        Args:
            ema_period: Period for the middle band EMA (default 20)
            atr_period: Period for ATR calculation (default 10)
            multiplier: Multiplier for ATR to create bands (default 2.0)
            price_key: Key to extract prices from data points
        """
        params = {
            'ema_period': ema_period,
            'atr_period': atr_period,
            'multiplier': multiplier,
            'price_key': price_key
        }
        super().__init__(f"Keltner_{ema_period}_{atr_period}_{multiplier}", params)
        
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.price_key = price_key
        
        # Initialize state
        self.state = {
            # EMA state
            'last_ema': None,
            'ema_alpha': 2.0 / (ema_period + 1),
            'ema_count': 0,
            
            # ATR state
            'tr_values': [],
            'last_atr': None,
            'atr_count': 0,
            'prev_high': None,
            'prev_low': None,
            'prev_close': None
        }
    
    def update(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update Keltner Channels with a new data point
        
        Args:
            data_point: New data point
            
        Returns:
            Dictionary with middle, upper, and lower band values
        """
        if not self.is_initialized:
            return {'middle': None, 'upper': None, 'lower': None}
        
        # Need all required price data
        if not all(k in data_point for k in [self.price_key, 'high', 'low']):
            return {'middle': None, 'upper': None, 'lower': None}
        
        # Update timestamp
        if 'timestamp' in data_point:
            self.last_update_time = data_point['timestamp']
        
        price = data_point[self.price_key]
        high = data_point['high']
        low = data_point['low']
        
        # Update EMA calculation
        if self.state['last_ema'] is None:
            # First EMA value is the price itself
            self.state['last_ema'] = price
            self.state['ema_count'] = 1
        else:
            # Update EMA recursively
            self.state['last_ema'] = (
                price * self.state['ema_alpha'] + 
                self.state['last_ema'] * (1 - self.state['ema_alpha'])
            )
            self.state['ema_count'] += 1
        
        # Update ATR calculation
        prev_high = self.state.get('prev_high')
        prev_low = self.state.get('prev_low')
        prev_close = self.state.get('prev_close')
        
        if prev_high is not None and prev_low is not None and prev_close is not None:
            # Calculate True Range
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            
            self.state['atr_count'] += 1
            
            # Initial collection period for ATR
            if self.state['atr_count'] <= self.atr_period:
                self.state['tr_values'].append(tr)
                
                # After atr_period points, calculate initial ATR
                if self.state['atr_count'] == self.atr_period:
                    self.state['last_atr'] = sum(self.state['tr_values']) / self.atr_period
            else:
                # Subsequent calculations use Wilder's smoothing
                last_atr = self.state['last_atr']
                # ATR = [(Prior ATR × (period-1)) + Current TR] / period
                new_atr = ((last_atr * (self.atr_period - 1)) + tr) / self.atr_period
                self.state['last_atr'] = new_atr
        
        # Update previous values for next TR calculation
        self.state['prev_high'] = high
        self.state['prev_low'] = low
        self.state['prev_close'] = price
        
        # Calculate Keltner Channels
        middle = self.state['last_ema']
        atr = self.state['last_atr']
        
        # Need both EMA and ATR to calculate channels
        if middle is None or atr is None:
            return {'middle': middle, 'upper': None, 'lower': None}
        
        # Calculate upper and lower bands
        upper = middle + (self.multiplier * atr)
        lower = middle - (self.multiplier * atr)
        
        return {
            'middle': middle,
            'upper': upper,
            'lower': lower,
            'width': (upper - lower) / middle if middle != 0 else None
        }
    
    def initialize(self, historical_data: List[Dict[str, Any]]) -> None:
        """
        Initialize Keltner Channels with historical data
        
        Args:
            historical_data: List of historical data points
        """
        self.reset()
        
        if not historical_data:
            return
        
        # Need at least max(ema_period, atr_period) + 1 data points
        min_required = max(self.ema_period, self.atr_period) + 1
        if len(historical_data) < min_required:
            self.logger.warning(
                f"Insufficient historical data for {self.name}. "
                f"Need at least {min_required} points."
            )
            return
        
        # Process historical data
        prices = []
        high_prices = []
        low_prices = []
        close_prices = []
        tr_values = []
        
        for i, data_point in enumerate(historical_data):
            if not all(k in data_point for k in [self.price_key, 'high', 'low']):
                continue
                
            price = data_point[self.price_key]
            high = data_point['high']
            low = data_point['low']
            
            prices.append(price)
            high_prices.append(high)
            low_prices.append(low)
            close_prices.append(price)
            
            # Calculate TR values
            if i > 0:
                prev_high = high_prices[i-1]
                prev_low = low_prices[i-1]
                prev_close = close_prices[i-1]
                
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                tr_values.append(tr)
        
        # Calculate initial EMA (use SMA for first EMA value)
        if len(prices) >= self.ema_period:
            # Calculate first EMA as SMA
            initial_sma = sum(prices[-self.ema_period:]) / self.ema_period
            ema = initial_sma
            
            # Calculate final EMA value
            for price in prices[-self.ema_period+1:]:
                ema = price * self.state['ema_alpha'] + ema * (1 - self.state['ema_alpha'])
            
            self.state['last_ema'] = ema
            self.state['ema_count'] = self.ema_period
        
        # Calculate initial ATR
        if len(tr_values) >= self.atr_period:
            # Use most recent atr_period values
            recent_tr = tr_values[-self.atr_period:]
            initial_atr = sum(recent_tr) / self.atr_period
            
            self.state['last_atr'] = initial_atr
            self.state['atr_count'] = self.atr_period
            self.state['tr_values'] = recent_tr
        
        # Store last price values for next update
        if historical_data:
            last_point = historical_data[-1]
            if all(k in last_point for k in ['high', 'low', self.price_key]):
                self.state['prev_high'] = last_point['high']
                self.state['prev_low'] = last_point['low']
                self.state['prev_close'] = last_point[self.price_key]
            
            if 'timestamp' in last_point:
                self.last_update_time = last_point['timestamp']
        
        self.is_initialized = (
            self.state['last_ema'] is not None and 
            self.state['last_atr'] is not None
        )
