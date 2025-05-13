"""
Incremental Technical Indicator Calculator

This module provides classes for incrementally calculating technical indicators
with minimal computational overhead, optimized for low-latency environments.
"""
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
import logging
import json


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@dataclass
class IndicatorState:
    """Base class for storing the state required for incremental indicator updates"""
    name: str
    parameters: Dict[str, Any]
    last_value: Optional[float] = None
    is_ready: bool = False

    def update(self, new_data: Dict[str, float]) ->Optional[float]:
        """
        Update the indicator with new data
        
        Args:
            new_data: Dictionary containing latest price data
                     (e.g., {'close': 1.2345, 'high': 1.2355, ...})
        
        Returns:
            Updated indicator value or None if not enough data
        """
        raise NotImplementedError('Subclasses must implement update method')

    def reset(self) ->None:
        """Reset the internal state"""
        raise NotImplementedError('Subclasses must implement reset method')


class SMAState(IndicatorState):
    """State for Simple Moving Average incremental calculation"""

    def __init__(self, window_size: int, price_type: str='close'):
    """
      init  .
    
    Args:
        window_size: Description of window_size
        price_type: Description of price_type
    
    """

        super().__init__(name=f'SMA_{window_size}', parameters={
            'window_size': window_size, 'price_type': price_type})
        self.window_size = window_size
        self.price_type = price_type
        self.values = []
        self.sum = 0.0

    def update(self, new_data: Dict[str, float]) ->Optional[float]:
    """
    Update.
    
    Args:
        new_data: Description of new_data
        float]: Description of float]
    
    Returns:
        Optional[float]: Description of return value
    
    """

        if self.price_type not in new_data:
            logging.warning(
                f'Price type {self.price_type} not found in data: {new_data}')
            return self.last_value
        price = new_data[self.price_type]
        self.values.append(price)
        self.sum += price
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        if len(self.values) == self.window_size:
            self.is_ready = True
            self.last_value = self.sum / self.window_size
            return self.last_value
        else:
            return None

    def reset(self) ->None:
        self.values = []
        self.sum = 0.0
        self.last_value = None
        self.is_ready = False


class EMAState(IndicatorState):
    """State for Exponential Moving Average incremental calculation"""

    def __init__(self, window_size: int, price_type: str='close', smoothing:
        float=2.0):
    """
      init  .
    
    Args:
        window_size: Description of window_size
        price_type: Description of price_type
        smoothing: Description of smoothing
    
    """

        super().__init__(name=f'EMA_{window_size}', parameters={
            'window_size': window_size, 'price_type': price_type,
            'smoothing': smoothing})
        self.window_size = window_size
        self.price_type = price_type
        self.smoothing = smoothing
        self.multiplier = smoothing / (window_size + 1)
        self.count = 0
        self.values = []

    def update(self, new_data: Dict[str, float]) ->Optional[float]:
    """
    Update.
    
    Args:
        new_data: Description of new_data
        float]: Description of float]
    
    Returns:
        Optional[float]: Description of return value
    
    """

        if self.price_type not in new_data:
            logging.warning(
                f'Price type {self.price_type} not found in data: {new_data}')
            return self.last_value
        price = new_data[self.price_type]
        self.count += 1
        if not self.is_ready:
            self.values.append(price)
            if len(self.values) == self.window_size:
                sma = sum(self.values) / self.window_size
                self.last_value = sma
                self.is_ready = True
            return self.last_value
        self.last_value = price * self.multiplier + self.last_value * (1 -
            self.multiplier)
        return self.last_value

    def reset(self) ->None:
    """
    Reset.
    
    """

        self.count = 0
        self.last_value = None
        self.is_ready = False
        self.values = []
        if not self.is_ready and self.count == self.window_size:
            self.is_ready = True
            self.last_value = price
            return self.last_value
        if self.is_ready:
            self.last_value = price * self.multiplier + self.last_value * (
                1 - self.multiplier)
            return self.last_value
        return None

    def reset(self) ->None:
        self.count = 0
        self.last_value = None
        self.is_ready = False


class RSIState(IndicatorState):
    """State for Relative Strength Index incremental calculation"""

    def __init__(self, window_size: int=14, price_type: str='close'):
    """
      init  .
    
    Args:
        window_size: Description of window_size
        price_type: Description of price_type
    
    """

        super().__init__(name=f'RSI_{window_size}', parameters={
            'window_size': window_size, 'price_type': price_type})
        self.window_size = window_size
        self.price_type = price_type
        self.gains = []
        self.losses = []
        self.prev_price = None
        self.avg_gain = None
        self.avg_loss = None
        self.count = 0

    def update(self, new_data: Dict[str, float]) ->Optional[float]:
    """
    Update.
    
    Args:
        new_data: Description of new_data
        float]: Description of float]
    
    Returns:
        Optional[float]: Description of return value
    
    """

        if self.price_type not in new_data:
            return self.last_value
        current_price = new_data[self.price_type]
        if self.prev_price is None:
            self.prev_price = current_price
            return None
        change = current_price - self.prev_price
        self.count += 1
        if change > 0:
            gain = change
            loss = 0
        else:
            gain = 0
            loss = abs(change)
        self.gains.append(gain)
        self.losses.append(loss)
        if len(self.gains) > self.window_size:
            self.gains.pop(0)
            self.losses.pop(0)
        if self.count == self.window_size + 1:
            self.avg_gain = sum(self.gains) / self.window_size
            self.avg_loss = sum(self.losses) / self.window_size
            self.is_ready = True
        elif self.is_ready:
            self.avg_gain = (self.avg_gain * (self.window_size - 1) + gain
                ) / self.window_size
            self.avg_loss = (self.avg_loss * (self.window_size - 1) + loss
                ) / self.window_size
        if self.is_ready:
            if self.avg_loss == 0:
                self.last_value = 100
            else:
                rs = (self.avg_gain / self.avg_loss if self.avg_loss > 0 else
                    float('inf'))
                self.last_value = 100 - 100 / (1 + rs)
        self.prev_price = current_price
        return self.last_value

    def reset(self) ->None:
    """
    Reset.
    
    """

        self.gains = []
        self.losses = []
        self.prev_price = None
        self.avg_gain = None
        self.avg_loss = None
        self.count = 0
        self.last_value = None
        self.is_ready = False


class MACDState(IndicatorState):
    """State for Moving Average Convergence Divergence incremental calculation"""

    def __init__(self, fast_period: int=12, slow_period: int=26,
        signal_period: int=9, price_type: str='close'):
    """
      init  .
    
    Args:
        fast_period: Description of fast_period
        slow_period: Description of slow_period
        signal_period: Description of signal_period
        price_type: Description of price_type
    
    """

        super().__init__(name=
            f'MACD_{fast_period}_{slow_period}_{signal_period}', parameters
            ={'fast_period': fast_period, 'slow_period': slow_period,
            'signal_period': signal_period, 'price_type': price_type})
        self.fast_ema = EMAState(fast_period, price_type)
        self.slow_ema = EMAState(slow_period, price_type)
        self.signal_ema = EMAState(signal_period, 'macd_line')
        self.macd_history = []
        self.signal_count = 0

    def update(self, new_data: Dict[str, float]) ->Dict[str, Optional[float]]:
    """
    Update.
    
    Args:
        new_data: Description of new_data
        float]: Description of float]
    
    Returns:
        Dict[str, Optional[float]]: Description of return value
    
    """

        fast_value = self.fast_ema.update(new_data)
        slow_value = self.slow_ema.update(new_data)
        result = {'macd_line': None, 'signal_line': None, 'histogram': None}
        if fast_value is not None and slow_value is not None:
            macd_line = fast_value - slow_value
            result['macd_line'] = macd_line
            signal_data = {'macd_line': macd_line}
            signal_value = self.signal_ema.update(signal_data)
            if signal_value is not None:
                result['signal_line'] = signal_value
                result['histogram'] = macd_line - signal_value
                self.is_ready = True
        self.last_value = result
        return result if self.is_ready else None

    def reset(self) ->None:
    """
    Reset.
    
    """

        self.fast_ema.reset()
        self.slow_ema.reset()
        self.signal_ema.reset()
        self.macd_history = []
        self.signal_count = 0
        self.last_value = None
        self.is_ready = False


class BollingerBandsState(IndicatorState):
    """State for Bollinger Bands incremental calculation"""

    def __init__(self, window_size: int=20, num_std: float=2.0, price_type:
        str='close'):
    """
      init  .
    
    Args:
        window_size: Description of window_size
        num_std: Description of num_std
        price_type: Description of price_type
    
    """

        super().__init__(name=f'BBANDS_{window_size}_{num_std}', parameters
            ={'window_size': window_size, 'num_std': num_std, 'price_type':
            price_type})
        self.window_size = window_size
        self.num_std = num_std
        self.price_type = price_type
        self.prices = []
        self.sum = 0.0
        self.sum_sq = 0.0

    def update(self, new_data: Dict[str, float]) ->Optional[Dict[str, float]]:
    """
    Update.
    
    Args:
        new_data: Description of new_data
        float]: Description of float]
    
    Returns:
        Optional[Dict[str, float]]: Description of return value
    
    """

        if self.price_type not in new_data:
            logging.warning(
                f'Price type {self.price_type} not found in data: {new_data}')
            return self.last_value
        price = new_data[self.price_type]
        self.prices.append(price)
        self.sum += price
        self.sum_sq += price * price
        if len(self.prices) > self.window_size:
            old_price = self.prices.pop(0)
            self.sum -= old_price
            self.sum_sq -= old_price * old_price
        if len(self.prices) == self.window_size:
            self.is_ready = True
            middle_band = self.sum / self.window_size
            variance = (self.sum_sq - self.sum ** 2 / self.window_size
                ) / self.window_size
            std_dev = max(0.0001, np.sqrt(variance))
            upper_band = middle_band + self.num_std * std_dev
            lower_band = middle_band - self.num_std * std_dev
            result = {'middle_band': middle_band, 'upper_band': upper_band,
                'lower_band': lower_band, 'bandwidth': (upper_band -
                lower_band) / middle_band if middle_band > 0 else 0}
            self.last_value = result
            return result
        else:
            return None

    def reset(self) ->None:
        self.prices = []
        self.sum = 0.0
        self.sum_sq = 0.0
        self.last_value = None
        self.is_ready = False


class ATRState(IndicatorState):
    """State for Average True Range incremental calculation"""

    def __init__(self, window_size: int=14):
    """
      init  .
    
    Args:
        window_size: Description of window_size
    
    """

        super().__init__(name=f'ATR_{window_size}', parameters={
            'window_size': window_size})
        self.window_size = window_size
        self.tr_values = []
        self.prev_close = None
        self.atr = None
        self.count = 0

    def update(self, new_data: Dict[str, float]) ->Optional[float]:
    """
    Update.
    
    Args:
        new_data: Description of new_data
        float]: Description of float]
    
    Returns:
        Optional[float]: Description of return value
    
    """

        required_fields = ['high', 'low', 'close']
        for field in required_fields:
            if field not in new_data:
                logging.warning(
                    f'Required field {field} not found in data: {new_data}')
                return self.last_value
        high = new_data['high']
        low = new_data['low']
        close = new_data['close']
        if self.prev_close is not None:
            tr = max(high - low, abs(high - self.prev_close), abs(low -
                self.prev_close))
        else:
            tr = high - low
        self.tr_values.append(tr)
        self.count += 1
        if len(self.tr_values) > self.window_size:
            self.tr_values.pop(0)
        if self.count >= self.window_size:
            if self.atr is None:
                self.atr = sum(self.tr_values) / self.window_size
            else:
                self.atr = (self.atr * (self.window_size - 1) + tr
                    ) / self.window_size
            self.is_ready = True
            self.last_value = self.atr
        self.prev_close = close
        return self.last_value if self.is_ready else None

    def reset(self) ->None:
    """
    Reset.
    
    """

        self.tr_values = []
        self.prev_close = None
        self.atr = None
        self.count = 0
        self.last_value = None
        self.is_ready = False


class IncrementalIndicatorManager:
    """
    Manager class for maintaining and updating multiple incremental indicators
    """

    def __init__(self):
        self.indicators: Dict[str, IndicatorState] = {}
        self.logger = logging.getLogger(__name__)

    def add_indicator(self, indicator: IndicatorState) ->None:
        """
        Add an indicator to the manager
        
        Args:
            indicator: Initialized indicator state object
        """
        self.indicators[indicator.name] = indicator
        self.logger.info(
            f'Added indicator: {indicator.name} with parameters {indicator.parameters}'
            )

    def remove_indicator(self, indicator_name: str) ->bool:
        """
        Remove an indicator from the manager
        
        Args:
            indicator_name: Name of the indicator to remove
            
        Returns:
            True if removed successfully, False otherwise
        """
        if indicator_name in self.indicators:
            del self.indicators[indicator_name]
            self.logger.info(f'Removed indicator: {indicator_name}')
            return True
        return False

    @with_exception_handling
    def update_all(self, new_data: Dict[str, float]) ->Dict[str, Any]:
        """
        Update all indicators with new price data
        
        Args:
            new_data: Dictionary containing latest price data
            
        Returns:
            Dictionary with updated indicator values
        """
        results = {}
        for name, indicator in self.indicators.items():
            try:
                value = indicator.update(new_data)
                if value is not None:
                    results[name] = value
            except Exception as e:
                self.logger.error(f'Error updating indicator {name}: {str(e)}')
        return results

    def reset_all(self) ->None:
        """Reset all indicators to their initial state"""
        for indicator in self.indicators.values():
            indicator.reset()
        self.logger.info('Reset all indicators')

    def get_latest_values(self) ->Dict[str, Any]:
        """
        Get the latest values for all ready indicators
        
        Returns:
            Dictionary mapping indicator names to their latest values
        """
        return {name: indicator.last_value for name, indicator in self.
            indicators.items() if indicator.is_ready}
