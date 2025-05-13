"""
Degraded Mode Indicators Module

This module provides fallback implementations for basic technical indicators
that can be used when the feature-store-service is unavailable.

These implementations prioritize reliability and performance over advanced features,
serving as resilient alternatives during dependency outages.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime
from core_foundations.resilience.degraded_mode import fallback_for
from analysis_engine.resilience.degraded_mode_strategies import AnalysisEngineDegradedMode
logger = logging.getLogger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class DegradedIndicator:
    """Base class for all degraded mode indicators"""

    def __init__(self, name: str, params: Dict[str, Any]=None):
        """
        Initialize degraded indicator
        
        Args:
            name: Indicator name
            params: Indicator parameters
        """
        self.name = name
        self.params = params or {}

    def calculate(self, data: pd.DataFrame) ->pd.DataFrame:
        """
        Calculate indicator values
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with indicator values added
        """
        raise NotImplementedError('Subclasses must implement calculate()')

    def _validate_data(self, data: pd.DataFrame, required_columns: List[str]
        ) ->bool:
        """Validate that DataFrame has required columns"""
        if data is None or data.empty:
            logger.warning(f'Empty data provided to {self.name}')
            return False
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            logger.warning(
                f'Missing required columns for {self.name}: {missing}')
            return False
        return True


@fallback_for('feature-store', 'get_moving_average')
class DegradedMovingAverage(DegradedIndicator):
    """Simple Moving Average - Degraded Mode Implementation"""

    def __init__(self, window: int=14, price_column: str='close'):
        """
        Initialize Simple Moving Average indicator
        
        Args:
            window: Number of periods for moving average
            price_column: Column to use for price data
        """
        params = {'window': window, 'price_column': price_column}
        super().__init__('DegradedMA', params)

    @with_exception_handling
    def calculate(self, data: pd.DataFrame) ->pd.DataFrame:
        """Calculate Simple Moving Average"""
        result = data.copy()
        if not self._validate_data(data, [self.params['price_column']]):
            return result
        window = self.params['window']
        price_col = self.params['price_column']
        try:
            result[f'degraded_ma_{window}'] = result[price_col].rolling(window
                =window, min_periods=1).mean()
            AnalysisEngineDegradedMode.update_cache('indicators',
                f'ma_{price_col}_{window}', result[f'degraded_ma_{window}']
                .iloc[-1] if not result.empty else None)
        except Exception as e:
            logger.error(f'Error calculating degraded MA: {str(e)}')
        return result


@fallback_for('feature-store', 'get_exponential_moving_average')
class DegradedExponentialMovingAverage(DegradedIndicator):
    """Exponential Moving Average - Degraded Mode Implementation"""

    def __init__(self, window: int=14, price_column: str='close', smoothing:
        float=2.0):
        """
        Initialize Exponential Moving Average indicator
        
        Args:
            window: Number of periods for EMA
            price_column: Column to use for price data
            smoothing: Smoothing factor (default is 2.0 which is standard EMA)
        """
        params = {'window': window, 'price_column': price_column,
            'smoothing': smoothing}
        super().__init__('DegradedEMA', params)

    @with_exception_handling
    def calculate(self, data: pd.DataFrame) ->pd.DataFrame:
        """Calculate Exponential Moving Average"""
        result = data.copy()
        if not self._validate_data(data, [self.params['price_column']]):
            return result
        window = self.params['window']
        price_col = self.params['price_column']
        try:
            result[f'degraded_ema_{window}'] = result[price_col].ewm(span=
                window, adjust=False).mean()
            AnalysisEngineDegradedMode.update_cache('indicators',
                f'ema_{price_col}_{window}', result[
                f'degraded_ema_{window}'].iloc[-1] if not result.empty else
                None)
        except Exception as e:
            logger.error(f'Error calculating degraded EMA: {str(e)}')
        return result


@fallback_for('feature-store', 'get_relative_strength_index')
class DegradedRelativeStrengthIndex(DegradedIndicator):
    """Relative Strength Index - Degraded Mode Implementation"""

    def __init__(self, window: int=14, price_column: str='close'):
        """
        Initialize RSI indicator
        
        Args:
            window: Number of periods for RSI calculation
            price_column: Column to use for price data
        """
        params = {'window': window, 'price_column': price_column}
        super().__init__('DegradedRSI', params)

    @with_exception_handling
    def calculate(self, data: pd.DataFrame) ->pd.DataFrame:
        """Calculate Relative Strength Index"""
        result = data.copy()
        if not self._validate_data(data, [self.params['price_column']]) or len(
            data) < self.params['window']:
            return result
        window = self.params['window']
        price_col = self.params['price_column']
        try:
            delta = result[price_col].diff()
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = abs(loss)
            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()
            if len(gain) > window:
                for i in range(window, len(gain)):
                    avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (window - 1) +
                        gain.iloc[i]) / window
                    avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (window - 1) +
                        loss.iloc[i]) / window
            rs = avg_gain / avg_loss
            rsi = 100 - 100 / (1 + rs)
            rsi = rsi.replace([np.inf, -np.inf], 100)
            rsi = rsi.fillna(50)
            result[f'degraded_rsi_{window}'] = rsi
            AnalysisEngineDegradedMode.update_cache('indicators',
                f'rsi_{price_col}_{window}', result[
                f'degraded_rsi_{window}'].iloc[-1] if not result.empty else
                None)
        except Exception as e:
            logger.error(f'Error calculating degraded RSI: {str(e)}')
        return result


@fallback_for('feature-store', 'get_bollinger_bands')
class DegradedBollingerBands(DegradedIndicator):
    """Bollinger Bands - Degraded Mode Implementation"""

    def __init__(self, window: int=20, num_std: float=2.0, price_column:
        str='close'):
        """
        Initialize Bollinger Bands indicator
        
        Args:
            window: Number of periods for moving average
            num_std: Number of standard deviations for the bands
            price_column: Column to use for price data
        """
        params = {'window': window, 'num_std': num_std, 'price_column':
            price_column}
        super().__init__('DegradedBollingerBands', params)

    @with_exception_handling
    def calculate(self, data: pd.DataFrame) ->pd.DataFrame:
        """Calculate Bollinger Bands"""
        result = data.copy()
        if not self._validate_data(data, [self.params['price_column']]) or len(
            data) < self.params['window']:
            return result
        window = self.params['window']
        num_std = self.params['num_std']
        price_col = self.params['price_column']
        try:
            middle_band = result[price_col].rolling(window=window,
                min_periods=1).mean()
            std_dev = result[price_col].rolling(window=window, min_periods=1
                ).std()
            upper_band = middle_band + std_dev * num_std
            lower_band = middle_band - std_dev * num_std
            result[f'degraded_bb_middle_{window}'] = middle_band
            result[f'degraded_bb_upper_{window}'] = upper_band
            result[f'degraded_bb_lower_{window}'] = lower_band
            if not result.empty:
                last_values = {'middle': middle_band.iloc[-1], 'upper':
                    upper_band.iloc[-1], 'lower': lower_band.iloc[-1]}
                AnalysisEngineDegradedMode.update_cache('indicators',
                    f'bollinger_{price_col}_{window}_{num_std}', last_values)
        except Exception as e:
            logger.error(
                f'Error calculating degraded Bollinger Bands: {str(e)}')
        return result


@fallback_for('feature-store', 'get_macd')
class DegradedMACD(DegradedIndicator):
    """MACD - Degraded Mode Implementation"""

    def __init__(self, fast_period: int=12, slow_period: int=26,
        signal_period: int=9, price_column: str='close'):
        """
        Initialize MACD indicator
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            price_column: Column to use for price data
        """
        params = {'fast_period': fast_period, 'slow_period': slow_period,
            'signal_period': signal_period, 'price_column': price_column}
        super().__init__('DegradedMACD', params)

    @with_exception_handling
    def calculate(self, data: pd.DataFrame) ->pd.DataFrame:
        """Calculate MACD"""
        result = data.copy()
        if not self._validate_data(data, [self.params['price_column']]):
            return result
        fast_period = self.params['fast_period']
        slow_period = self.params['slow_period']
        signal_period = self.params['signal_period']
        price_col = self.params['price_column']
        try:
            fast_ema = result[price_col].ewm(span=fast_period, adjust=False
                ).mean()
            slow_ema = result[price_col].ewm(span=slow_period, adjust=False
                ).mean()
            macd_line = fast_ema - slow_ema
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean(
                )
            histogram = macd_line - signal_line
            result['degraded_macd_line'] = macd_line
            result['degraded_macd_signal'] = signal_line
            result['degraded_macd_histogram'] = histogram
            if not result.empty:
                last_values = {'macd': macd_line.iloc[-1], 'signal':
                    signal_line.iloc[-1], 'histogram': histogram.iloc[-1]}
                AnalysisEngineDegradedMode.update_cache('indicators',
                    f'macd_{price_col}_{fast_period}_{slow_period}_{signal_period}'
                    , last_values)
        except Exception as e:
            logger.error(f'Error calculating degraded MACD: {str(e)}')
        return result


def register_all_degraded_indicators():
    """Register all degraded mode indicators with the DegradedModeManager"""
    from core_foundations.resilience.degraded_mode import DegradedModeManager
    manager = DegradedModeManager.get_instance()
    manager.register_fallback_strategy('feature-store',
        'get_moving_average', DegradedMovingAverage)
    manager.register_fallback_strategy('feature-store',
        'get_exponential_moving_average', DegradedExponentialMovingAverage)
    manager.register_fallback_strategy('feature-store',
        'get_relative_strength_index', DegradedRelativeStrengthIndex)
    manager.register_fallback_strategy('feature-store',
        'get_bollinger_bands', DegradedBollingerBands)
    manager.register_fallback_strategy('feature-store', 'get_macd',
        DegradedMACD)
    logger.info('Registered all degraded mode indicators')
