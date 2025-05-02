"""
Extended Incremental Technical Indicators

This module extends the incremental indicators framework with additional technical indicators
optimized for low-latency environments.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Deque
from collections import deque
import logging

from feature_store_service.indicators.incremental_indicators import IndicatorState


class RSIState(IndicatorState):
    """
    State for Relative Strength Index incremental calculation
    
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    def __init__(self, window_size: int = 14, price_type: str = 'close'):
        super().__init__(
            name=f"RSI_{window_size}",
            parameters={"window_size": window_size, "price_type": price_type}
        )
        self.window_size = window_size
        self.price_type = price_type
        self.gains = []
        self.losses = []
        self.prev_price = None
        self.avg_gain = None
        self.avg_loss = None
    
    def update(self, new_data: Dict[str, float]) -> Optional[float]:
        if self.price_type not in new_data:
            logging.warning(f"Price type {self.price_type} not found in data: {new_data}")
            return self.last_value
        
        current_price = new_data[self.price_type]
        
        # Need at least two prices to calculate change
        if self.prev_price is None:
            self.prev_price = current_price
            return None
        
        # Calculate price change and separate into gain and loss
        change = current_price - self.prev_price
        gain = max(0, change)
        loss = max(0, -change)
        
        self.gains.append(gain)
        self.losses.append(abs(loss))  # Store loss as positive value
        
        # Update previous price for next calculation
        self.prev_price = current_price
        
        # Check if we have enough data
        if len(self.gains) < self.window_size:
            return None
        
        # Remove oldest values if we exceed window size
        if len(self.gains) > self.window_size:
            self.gains.pop(0)
            self.losses.pop(0)
        
        # Calculate average gain and average loss
        if self.avg_gain is None or self.avg_loss is None:
            # First calculation - simple averages
            self.avg_gain = sum(self.gains) / self.window_size
            self.avg_loss = sum(self.losses) / self.window_size
        else:
            # Subsequent calculations - smoothed averages
            self.avg_gain = (self.avg_gain * (self.window_size - 1) + self.gains[-1]) / self.window_size
            self.avg_loss = (self.avg_loss * (self.window_size - 1) + self.losses[-1]) / self.window_size
        
        # Calculate RS and RSI
        if self.avg_loss == 0:
            rsi = 100.0  # No losses means RSI=100
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        
        self.is_ready = True
        self.last_value = rsi
        return rsi
    
    def reset(self) -> None:
        self.gains = []
        self.losses = []
        self.prev_price = None
        self.avg_gain = None
        self.avg_loss = None
        self.last_value = None
        self.is_ready = False


class BollingerBandsState(IndicatorState):
    """
    State for Bollinger Bands incremental calculation
    
    Middle Band = SMA
    Upper Band = Middle Band + (Standard Deviation * Multiplier)
    Lower Band = Middle Band - (Standard Deviation * Multiplier)
    """
    def __init__(self, window_size: int = 20, num_std: float = 2.0, price_type: str = 'close'):
        super().__init__(
            name=f"BBANDS_{window_size}_{num_std}",
            parameters={"window_size": window_size, "num_std": num_std, "price_type": price_type}
        )
        self.window_size = window_size
        self.num_std = num_std
        self.price_type = price_type
        self.prices = deque(maxlen=window_size)
        self.sma = None
        self.upper_band = None
        self.lower_band = None
    
    def update(self, new_data: Dict[str, float]) -> Optional[Dict[str, float]]:
        if self.price_type not in new_data:
            logging.warning(f"Price type {self.price_type} not found in data: {new_data}")
            return None
        
        price = new_data[self.price_type]
        self.prices.append(price)
        
        # Need full window for calculation
        if len(self.prices) < self.window_size:
            return None
        
        # Calculate SMA (middle band)
        self.sma = sum(self.prices) / self.window_size
        
        # Calculate standard deviation
        variance = sum((x - self.sma) ** 2 for x in self.prices) / len(self.prices)
        std_dev = np.sqrt(variance)
        
        # Calculate upper and lower bands
        self.upper_band = self.sma + (std_dev * self.num_std)
        self.lower_band = self.sma - (std_dev * self.num_std)
        
        # For Bollinger Bands, we return a dictionary with all three bands
        result = {
            'middle': self.sma,
            'upper': self.upper_band,
            'lower': self.lower_band,
            'bandwidth': (self.upper_band - self.lower_band) / self.sma if self.sma != 0 else 0,
            'percent_b': (price - self.lower_band) / (self.upper_band - self.lower_band) if (self.upper_band - self.lower_band) != 0 else 0.5
        }
        
        self.is_ready = True
        self.last_value = result
        return result
    
    def reset(self) -> None:
        self.prices.clear()
        self.sma = None
        self.upper_band = None
        self.lower_band = None
        self.last_value = None
        self.is_ready = False


class MACDState(IndicatorState):
    """
    State for Moving Average Convergence Divergence incremental calculation
    
    MACD Line = Fast EMA - Slow EMA
    Signal Line = EMA of MACD Line
    Histogram = MACD Line - Signal Line
    """
    def __init__(
        self, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9, 
        price_type: str = 'close'
    ):
        super().__init__(
            name=f"MACD_{fast_period}_{slow_period}_{signal_period}",
            parameters={
                "fast_period": fast_period, 
                "slow_period": slow_period, 
                "signal_period": signal_period,
                "price_type": price_type
            }
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.price_type = price_type
        
        # For fast EMA calculation
        self.fast_count = 0
        self.fast_values = []
        self.fast_multiplier = 2.0 / (fast_period + 1)
        self.fast_ema = None
        
        # For slow EMA calculation
        self.slow_count = 0
        self.slow_values = []
        self.slow_multiplier = 2.0 / (slow_period + 1)
        self.slow_ema = None
        
        # For signal line calculation
        self.macd_values = []
        self.signal_multiplier = 2.0 / (signal_period + 1)
        self.signal_line = None
    
    def update(self, new_data: Dict[str, float]) -> Optional[Dict[str, float]]:
        if self.price_type not in new_data:
            logging.warning(f"Price type {self.price_type} not found in data: {new_data}")
            return None
        
        price = new_data[self.price_type]
        
        # Update fast EMA
        if self.fast_ema is None:
            # Initialize with SMA
            self.fast_values.append(price)
            self.fast_count += 1
            
            if self.fast_count == self.fast_period:
                self.fast_ema = sum(self.fast_values) / self.fast_period
        else:
            # Calculate EMA
            self.fast_ema = price * self.fast_multiplier + self.fast_ema * (1 - self.fast_multiplier)
        
        # Update slow EMA
        if self.slow_ema is None:
            # Initialize with SMA
            self.slow_values.append(price)
            self.slow_count += 1
            
            if self.slow_count == self.slow_period:
                self.slow_ema = sum(self.slow_values) / self.slow_period
        else:
            # Calculate EMA
            self.slow_ema = price * self.slow_multiplier + self.slow_ema * (1 - self.slow_multiplier)
        
        # Both EMAs need to be initialized before calculating MACD
        if self.fast_ema is None or self.slow_ema is None:
            return None
        
        # Calculate MACD
        macd_line = self.fast_ema - self.slow_ema
        self.macd_values.append(macd_line)
        
        # Need signal_period MACD values to calculate signal line
        if len(self.macd_values) < self.signal_period:
            return None
        
        # Calculate signal line
        if self.signal_line is None:
            self.signal_line = sum(self.macd_values[-self.signal_period:]) / self.signal_period
        else:
            self.signal_line = macd_line * self.signal_multiplier + self.signal_line * (1 - self.signal_multiplier)
        
        # Calculate histogram
        histogram = macd_line - self.signal_line
        
        result = {
            'macd': macd_line,
            'signal': self.signal_line,
            'histogram': histogram
        }
        
        self.is_ready = True
        self.last_value = result
        return result
    
    def reset(self) -> None:
        self.fast_count = 0
        self.fast_values = []
        self.fast_ema = None
        
        self.slow_count = 0
        self.slow_values = []
        self.slow_ema = None
        
        self.macd_values = []
        self.signal_line = None
        
        self.last_value = None
        self.is_ready = False


class ATRState(IndicatorState):
    """
    State for Average True Range incremental calculation
    
    True Range = max(high - low, abs(high - previous close), abs(low - previous close))
    ATR = SMA or EMA of True Range values
    """
    def __init__(self, window_size: int = 14, use_ema: bool = False):
        super().__init__(
            name=f"ATR_{window_size}{'_EMA' if use_ema else ''}",
            parameters={"window_size": window_size, "use_ema": use_ema}
        )
        self.window_size = window_size
        self.use_ema = use_ema
        self.tr_values = []
        self.atr = None
        self.prev_close = None
        self.multiplier = 2.0 / (window_size + 1) if use_ema else None
    
    def update(self, new_data: Dict[str, float]) -> Optional[float]:
        # ATR requires high, low, and close prices
        if not all(k in new_data for k in ('high', 'low', 'close')):
            logging.warning("Missing required price data for ATR calculation")
            return self.last_value
        
        high = new_data['high']
        low = new_data['low']
        close = new_data['close']
        
        # First point needs previous close for TR calculation
        if self.prev_close is None:
            self.prev_close = close
            return None
        
        # Calculate True Range
        tr = max(
            high - low,
            abs(high - self.prev_close),
            abs(low - self.prev_close)
        )
        
        # Update previous close for next calculation
        self.prev_close = close
        
        # Store TR value
        self.tr_values.append(tr)
        
        # Remove oldest value if we exceed window size
        if len(self.tr_values) > self.window_size:
            self.tr_values.pop(0)
        
        # Need full window for initial calculation
        if len(self.tr_values) < self.window_size:
            return None
        
        # Calculate ATR
        if self.atr is None:
            # First ATR value is simple average of TR values
            self.atr = sum(self.tr_values) / self.window_size
        else:
            if self.use_ema:
                # Exponential average method
                self.atr = tr * self.multiplier + self.atr * (1 - self.multiplier)
            else:
                # Simple moving average method
                self.atr = sum(self.tr_values) / self.window_size
        
        self.is_ready = True
        self.last_value = self.atr
        return self.atr
    
    def reset(self) -> None:
        self.tr_values = []
        self.atr = None
        self.prev_close = None
        self.last_value = None
        self.is_ready = False


class StochasticState(IndicatorState):
    """
    State for Stochastic Oscillator incremental calculation
    
    %K = 100 * ((close - lowest low) / (highest high - lowest low))
    %D = Simple Moving Average of %K
    """
    def __init__(
        self, 
        k_period: int = 14, 
        d_period: int = 3, 
        slowing: int = 3
    ):
        super().__init__(
            name=f"STOCH_{k_period}_{d_period}_{slowing}",
            parameters={"k_period": k_period, "d_period": d_period, "slowing": slowing}
        )
        self.k_period = k_period
        self.d_period = d_period
        self.slowing = slowing
        
        self.highs = deque(maxlen=k_period)
        self.lows = deque(maxlen=k_period)
        self.closes = deque(maxlen=slowing)
        self.k_values = deque(maxlen=d_period)
    
    def update(self, new_data: Dict[str, float]) -> Optional[Dict[str, float]]:
        # Stochastic requires high, low, and close prices
        if not all(k in new_data for k in ('high', 'low', 'close')):
            logging.warning("Missing required price data for Stochastic calculation")
            return self.last_value
        
        high = new_data['high']
        low = new_data['low']
        close = new_data['close']
        
        # Add prices to the window
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        
        # Need full window for calculation
        if len(self.highs) < self.k_period:
            return None
        
        # Calculate raw %K (without slowing)
        lowest_low = min(self.lows)
        highest_high = max(self.highs)
        
        # Prevent division by zero
        if highest_high == lowest_low:
            raw_k = 50.0  # Middle value when range is zero
        else:
            raw_k = 100.0 * ((close - lowest_low) / (highest_high - lowest_low))
        
        # Apply slowing if required
        if self.slowing > 1:
            # Need enough closes for slowing
            if len(self.closes) < self.slowing:
                return None
            
            # Calculate slowed %K
            k = sum(self.closes) / self.slowing
        else:
            k = raw_k
        
        # Store %K for %D calculation
        self.k_values.append(k)
        
        # Calculate %D
        if len(self.k_values) < self.d_period:
            d = None
        else:
            d = sum(self.k_values) / self.d_period
        
        result = {
            'k': k,
            'd': d
        }
        
        self.is_ready = True
        self.last_value = result
        return result
    
    def reset(self) -> None:
        self.highs.clear()
        self.lows.clear()
        self.closes.clear()
        self.k_values.clear()
        self.last_value = None
        self.is_ready = False
