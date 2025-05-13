"""
Oscillator Incremental Indicators

This module provides incremental implementations of various oscillator indicators
optimized for memory usage and processing efficiency.
"""
from typing import Dict, List, Optional, Any
import numpy as np
from core.base_6 import (
    StatefulIndicator,
    RecursiveIndicator,
    CompositeIndicator,
    IncrementalIndicator
)
from core.moving_averages_3 import EMA


class RSI(StatefulIndicator):
    """
    Relative Strength Index - Incremental Implementation
    
    RSI measures the speed and magnitude of price movements, indicating
    overbought or oversold conditions.
    """
    
    def __init__(self, window_size: int = 14, price_key: str = 'close'):
        """
        Initialize RSI indicator
        
        Args:
            window_size: Lookback period for RSI (default 14)
            price_key: Key to extract prices from data points
        """
        params = {'window_size': window_size, 'price_key': price_key}
        super().__init__(f"RSI_{window_size}", params)
        self.window_size = window_size
        self.price_key = price_key
        
        # Initialize state
        self.state = {
            'avg_gain': None,
            'avg_loss': None,
            'prev_price': None,
            'price_changes': [],
            'values_seen': 0
        }
    
    def update(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update RSI with new data point
        
        Args:
            data_point: New price data
            
        Returns:
            Updated RSI value
        """
        if not self.is_initialized:
            return {'value': None}
        
        if self.price_key not in data_point:
            return {'value': None}
        
        current_price = data_point[self.price_key]
        prev_price = self.state.get('prev_price')
        
        # Need a previous price to calculate change
        if prev_price is None:
            self.state['prev_price'] = current_price
            return {'value': None}
        
        # Calculate price change
        price_change = current_price - prev_price
        self.state['prev_price'] = current_price
        self.state['values_seen'] += 1
        
        # First period needs to collect changes
        if self.state['values_seen'] <= self.window_size:
            self.state['price_changes'].append(price_change)
            
            # After collecting enough changes, calculate initial averages
            if self.state['values_seen'] == self.window_size:
                gains = [max(0, change) for change in self.state['price_changes']]
                losses = [abs(min(0, change)) for change in self.state['price_changes']]
                
                self.state['avg_gain'] = sum(gains) / self.window_size
                self.state['avg_loss'] = sum(losses) / self.window_size
                
                # Calculate RSI
                rs = self._calculate_rs()
                rsi = 100 - (100 / (1 + rs)) if rs is not None else None
                
                return {'value': rsi}
            
            return {'value': None}
        
        # For subsequent periods, use the Wilder's smoothing method
        alpha = 1 / self.window_size
        current_gain = max(0, price_change)
        current_loss = abs(min(0, price_change))
        
        # Update average gain and loss
        self.state['avg_gain'] = (self.state['avg_gain'] * (self.window_size - 1) + current_gain) / self.window_size
        self.state['avg_loss'] = (self.state['avg_loss'] * (self.window_size - 1) + current_loss) / self.window_size
        
        # Calculate RSI
        rs = self._calculate_rs()
        rsi = 100 - (100 / (1 + rs)) if rs is not None else None
        
        return {'value': rsi}
    
    def _calculate_rs(self) -> Optional[float]:
        """Calculate Relative Strength value"""
        avg_gain = self.state.get('avg_gain')
        avg_loss = self.state.get('avg_loss')
        
        if avg_gain is None or avg_loss is None:
            return None
        
        # Avoid division by zero
        if avg_loss == 0:
            return float('inf')
            
        return avg_gain / avg_loss
    
    def initialize(self, historical_data: List[Dict[str, Any]]) -> None:
        """
        Initialize RSI with historical data
        
        Args:
            historical_data: List of historical data points
        """
        self.reset()
        
        if not historical_data:
            return
        
        # Need at least window_size + 1 values for a proper initialization
        if len(historical_data) <= self.window_size:
            self.logger.warning(
                f"Insufficient historical data for {self.name}. "
                f"Need at least {self.window_size + 1} points."
            )
            return
        
        # Extract prices and calculate changes
        prices = []
        for data_point in historical_data:
            if self.price_key in data_point:
                prices.append(data_point[self.price_key])
                if 'timestamp' in data_point:
                    self.last_update_time = data_point['timestamp']
        
        # Calculate price changes
        price_changes = []
        for i in range(1, len(prices)):
            price_changes.append(prices[i] - prices[i-1])
        
        # We need at least window_size price changes
        if len(price_changes) < self.window_size:
            return
        
        # Keep only the window_size most recent changes for initial calculation
        changes_for_init = price_changes[-self.window_size:]
        
        # Calculate initial averages
        gains = [max(0, change) for change in changes_for_init]
        losses = [abs(min(0, change)) for change in changes_for_init]
        
        self.state['avg_gain'] = sum(gains) / self.window_size
        self.state['avg_loss'] = sum(losses) / self.window_size
        self.state['prev_price'] = prices[-1]
        self.state['values_seen'] = self.window_size
        
        self.is_initialized = True


class Stochastic(StatefulIndicator):
    """
    Stochastic Oscillator - Incremental Implementation
    
    Stochastic oscillator is a momentum indicator comparing the closing price
    to its price range over a specific period.
    """
    
    def __init__(
        self, 
        k_period: int = 14, 
        d_period: int = 3,
        d_ma_type: str = 'simple'  # 'simple' or 'exponential'
    ):
        """
        Initialize Stochastic Oscillator
        
        Args:
            k_period: Lookback period for %K line (default 14)
            d_period: Periods for %D line (default 3)
            d_ma_type: Type of moving average for %D ('simple' or 'exponential')
        """
        params = {'k_period': k_period, 'd_period': d_period, 'd_ma_type': d_ma_type}
        super().__init__(f"Stochastic_{k_period}_{d_period}", params)
        self.k_period = k_period
        self.d_period = d_period
        self.d_ma_type = d_ma_type
        
        # Initialize state with price window
        self.state = {
            'high_prices': [],
            'low_prices': [],
            'close_prices': [],
            'k_values': [],
            'd_value': None,
        }
    
    def update(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update Stochastic with new data point
        
        Args:
            data_point: New price data including high, low, and close
            
        Returns:
            Updated Stochastic values (%K and %D)
        """
        if not self.is_initialized:
            return {'k': None, 'd': None}
        
        # Need high, low and close prices
        if 'high' not in data_point or 'low' not in data_point or 'close' not in data_point:
            return {'k': None, 'd': None}
        
        # Update timestamp
        if 'timestamp' in data_point:
            self.last_update_time = data_point['timestamp']
        
        # Extract prices
        high_price = data_point['high']
        low_price = data_point['low']
        close_price = data_point['close']
        
        # Update price windows
        self.state['high_prices'].append(high_price)
        self.state['low_prices'].append(low_price)
        self.state['close_prices'].append(close_price)
        
        # Keep only k_period prices
        if len(self.state['high_prices']) > self.k_period:
            self.state['high_prices'].pop(0)
            self.state['low_prices'].pop(0)
            self.state['close_prices'].pop(0)
        
        # Need a full window to calculate K
        if len(self.state['high_prices']) < self.k_period:
            return {'k': None, 'd': None}
        
        # Calculate %K
        highest_high = max(self.state['high_prices'])
        lowest_low = min(self.state['low_prices'])
        
        # Avoid division by zero
        if highest_high == lowest_low:
            k_value = 50.0  # Default to midpoint
        else:
            k_value = 100 * (close_price - lowest_low) / (highest_high - lowest_low)
        
        # Store K value for D calculation
        self.state['k_values'].append(k_value)
        
        # Keep only d_period K values
        if len(self.state['k_values']) > self.d_period:
            self.state['k_values'].pop(0)
        
        # Calculate %D based on MA type
        d_value = None
        
        if len(self.state['k_values']) == self.d_period:
            if self.d_ma_type == 'simple':
                d_value = sum(self.state['k_values']) / self.d_period
            else:  # exponential
                # Only calculate EMA if we have a previous value
                if self.state['d_value'] is not None:
                    alpha = 2 / (self.d_period + 1)
                    d_value = k_value * alpha + self.state['d_value'] * (1 - alpha)
                else:
                    # First calculation uses SMA
                    d_value = sum(self.state['k_values']) / self.d_period
        
        self.state['d_value'] = d_value
        
        return {'k': k_value, 'd': d_value}
    
    def initialize(self, historical_data: List[Dict[str, Any]]) -> None:
        """
        Initialize Stochastic with historical data
        
        Args:
            historical_data: List of historical data points
        """
        self.reset()
        
        if not historical_data:
            return
        
        # Need high, low and close prices
        for data_point in historical_data:
            if not all(k in data_point for k in ['high', 'low', 'close']):
                continue
            
            if 'timestamp' in data_point:
                self.last_update_time = data_point['timestamp']
            
            # Fill price windows
            self.state['high_prices'].append(data_point['high'])
            self.state['low_prices'].append(data_point['low'])
            self.state['close_prices'].append(data_point['close'])
            
            # Keep only the k_period most recent prices
            if len(self.state['high_prices']) > self.k_period:
                self.state['high_prices'].pop(0)
                self.state['low_prices'].pop(0)
                self.state['close_prices'].pop(0)
            
            # Once we have enough prices, calculate K values
            if len(self.state['high_prices']) == self.k_period:
                highest_high = max(self.state['high_prices'])
                lowest_low = min(self.state['low_prices'])
                close_price = self.state['close_prices'][-1]
                
                # Avoid division by zero
                if highest_high == lowest_low:
                    k_value = 50.0
                else:
                    k_value = 100 * (close_price - lowest_low) / (highest_high - lowest_low)
                
                self.state['k_values'].append(k_value)
                
                # Keep only d_period K values
                if len(self.state['k_values']) > self.d_period:
                    self.state['k_values'].pop(0)
                
                # Calculate %D when we have enough K values
                if len(self.state['k_values']) == self.d_period:
                    if self.d_ma_type == 'simple':
                        self.state['d_value'] = sum(self.state['k_values']) / self.d_period
                    else:  # exponential
                        if self.state['d_value'] is None:
                            self.state['d_value'] = sum(self.state['k_values']) / self.d_period
                        else:
                            alpha = 2 / (self.d_period + 1)
                            self.state['d_value'] = (k_value * alpha + 
                                                  self.state['d_value'] * (1 - alpha))
        
        self.is_initialized = len(self.state['high_prices']) == self.k_period


class ADX(StatefulIndicator):
    """
    Average Directional Index - Incremental Implementation
    
    ADX measures the strength of a trend (not direction) and can
    help determine if a market is trending or ranging.
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize ADX indicator
        
        Args:
            period: Lookback period for ADX calculation (default 14)
        """
        params = {'period': period}
        super().__init__(f"ADX_{period}", params)
        self.period = period
        
        # Initialize state with price window and DI values
        self.state = {
            'high_prices': [],
            'low_prices': [],
            'close_prices': [],
            'tr_values': [],  # True Range values
            'plus_dm_values': [],  # +DM values
            'minus_dm_values': [],  # -DM values
            'plus_di': None,  # +DI value
            'minus_di': None,  # -DI value
            'dx_values': [],  # Directional Index values
            'adx_value': None,  # ADX value
            'prev_high': None,
            'prev_low': None,
            'prev_close': None
        }
    
    def update(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update ADX with new data point
        
        Args:
            data_point: New price data including high, low, and close
            
        Returns:
            Updated ADX, +DI, and -DI values
        """
        if not self.is_initialized:
            return {'adx': None, 'plus_di': None, 'minus_di': None}
        
        # Need high, low and close prices
        if 'high' not in data_point or 'low' not in data_point or 'close' not in data_point:
            return {'adx': None, 'plus_di': None, 'minus_di': None}
        
        # Update timestamp
        if 'timestamp' in data_point:
            self.last_update_time = data_point['timestamp']
        
        # Extract prices
        high = data_point['high']
        low = data_point['low']
        close = data_point['close']
        
        prev_high = self.state.get('prev_high')
        prev_low = self.state.get('prev_low')
        prev_close = self.state.get('prev_close')
        
        # Need previous prices to calculate directional movement
        if prev_high is None or prev_low is None:
            self.state['prev_high'] = high
            self.state['prev_low'] = low
            self.state['prev_close'] = close
            return {'adx': None, 'plus_di': None, 'minus_di': None}
        
        # Calculate True Range
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        
        # Calculate Directional Movement
        plus_dm = high - prev_high if high - prev_high > prev_low - low and high - prev_high > 0 else 0
        minus_dm = prev_low - low if prev_low - low > high - prev_high and prev_low - low > 0 else 0
        
        # Update state
        self.state['tr_values'].append(tr)
        self.state['plus_dm_values'].append(plus_dm)
        self.state['minus_dm_values'].append(minus_dm)
        
        # Keep only period values
        if len(self.state['tr_values']) > self.period:
            self.state['tr_values'].pop(0)
            self.state['plus_dm_values'].pop(0)
            self.state['minus_dm_values'].pop(0)
        
        # Calculate smoothed values when we have enough data
        if len(self.state['tr_values']) == self.period:
            smoothed_tr = sum(self.state['tr_values'])
            smoothed_plus_dm = sum(self.state['plus_dm_values'])
            smoothed_minus_dm = sum(self.state['minus_dm_values'])
            
            # Calculate +DI and -DI (avoid division by zero)
            plus_di = 100 * smoothed_plus_dm / smoothed_tr if smoothed_tr > 0 else 0
            minus_di = 100 * smoothed_minus_dm / smoothed_tr if smoothed_tr > 0 else 0
            
            # Calculate DX and store for ADX calculation
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
            self.state['dx_values'].append(dx)
            
            # Keep only period DX values for ADX
            if len(self.state['dx_values']) > self.period:
                self.state['dx_values'].pop(0)
            
            # Calculate ADX as average of DX values
            adx = sum(self.state['dx_values']) / len(self.state['dx_values']) if self.state['dx_values'] else None
            
            # Store values for next calculation
            self.state['plus_di'] = plus_di
            self.state['minus_di'] = minus_di
            self.state['adx_value'] = adx
            
            # Update prev prices for next calculation
            self.state['prev_high'] = high
            self.state['prev_low'] = low
            self.state['prev_close'] = close
            
            return {
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di
            }
        
        # Not enough data yet
        self.state['prev_high'] = high
        self.state['prev_low'] = low
        self.state['prev_close'] = close
        
        return {'adx': None, 'plus_di': None, 'minus_di': None}
    
    def initialize(self, historical_data: List[Dict[str, Any]]) -> None:
        """
        Initialize ADX with historical data
        
        Args:
            historical_data: List of historical data points
        """
        self.reset()
        
        if not historical_data:
            return
        
        # Need at least period + 1 data points for proper initialization
        if len(historical_data) < self.period + 1:
            self.logger.warning(
                f"Insufficient historical data for {self.name}. "
                f"Need at least {self.period + 1} points."
            )
            return
        
        prev_high = None
        prev_low = None
        prev_close = None
        
        # Process each historical data point
        for i, data_point in enumerate(historical_data):
            if not all(k in data_point for k in ['high', 'low', 'close']):
                continue
                
            if 'timestamp' in data_point:
                self.last_update_time = data_point['timestamp']
            
            high = data_point['high']
            low = data_point['low']
            close = data_point['close']
            
            # First point just stores values
            if prev_high is None:
                prev_high = high
                prev_low = low
                prev_close = close
                continue
            
            # Calculate True Range
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            
            # Calculate Directional Movement
            plus_dm = high - prev_high if high - prev_high > prev_low - low and high - prev_high > 0 else 0
            minus_dm = prev_low - low if prev_low - low > high - prev_high and prev_low - low > 0 else 0
            
            # Store values
            self.state['tr_values'].append(tr)
            self.state['plus_dm_values'].append(plus_dm)
            self.state['minus_dm_values'].append(minus_dm)
            
            # Keep only period values
            if len(self.state['tr_values']) > self.period:
                self.state['tr_values'].pop(0)
                self.state['plus_dm_values'].pop(0)
                self.state['minus_dm_values'].pop(0)
            
            # Once we have enough TR/DM values, calculate DI and DX
            if len(self.state['tr_values']) == self.period:
                smoothed_tr = sum(self.state['tr_values'])
                smoothed_plus_dm = sum(self.state['plus_dm_values'])
                smoothed_minus_dm = sum(self.state['minus_dm_values'])
                
                plus_di = 100 * smoothed_plus_dm / smoothed_tr if smoothed_tr > 0 else 0
                minus_di = 100 * smoothed_minus_dm / smoothed_tr if smoothed_tr > 0 else 0
                
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
                self.state['dx_values'].append(dx)
                
                # Keep only period DX values
                if len(self.state['dx_values']) > self.period:
                    self.state['dx_values'].pop(0)
                
                # Store latest DI values
                self.state['plus_di'] = plus_di
                self.state['minus_di'] = minus_di
            
            # Update prev values for next iteration
            prev_high = high
            prev_low = low
            prev_close = close
        
        # Calculate ADX as average of DX values at the end
        if self.state['dx_values']:
            self.state['adx_value'] = sum(self.state['dx_values']) / len(self.state['dx_values'])
        
        # Store latest prices for future updates
        self.state['prev_high'] = prev_high
        self.state['prev_low'] = prev_low
        self.state['prev_close'] = prev_close
        
        self.is_initialized = True
