"""
Incremental Moving Averages Module.

This module provides implementations of moving averages that can be computed incrementally,
enabling efficient updates when new data arrives without recalculating the entire dataset.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from collections import deque
from core.base_incremental import IncrementalIndicator
from core_foundations.utils.logger import get_logger
logger = get_logger('feature-store-service.incremental-moving-averages')


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class IncrementalSMA(IncrementalIndicator):
    """
    Incremental Simple Moving Average (SMA).
    
    This class implements a Simple Moving Average that can be updated efficiently
    when new data arrives, maintaining a sliding window of prices to avoid
    recalculating the entire average.
    """

    def __init__(self, name: str='SMA', params: Dict[str, Any]=None):
        """
        Initialize the incremental SMA.
        
        Args:
            name: Base name for the SMA columns
            params: Dictionary of parameters for the SMA
                    - period: Period for the SMA (default: 14)
                    - column: Column to compute SMA for (default: "close")
        """
        params = params or {}
        self.period = params.get('period', 14)
        self.column = params.get('column', 'close')
        self.column_name = f'{name}_{self.period}'
        super().__init__(name, params)

    def initialize(self, data: pd.DataFrame) ->pd.DataFrame:
        """
        Initialize the SMA with historical data.
        
        Args:
            data: Historical OHLCV data
            
        Returns:
            DataFrame with SMA values added
        """
        if self.column not in data.columns:
            logger.error(f'Column {self.column} not found in data')
            self.is_initialized = False
            return data
        result = data.copy()
        result[self.column_name] = data[self.column].rolling(window=self.period
            ).mean()
        if len(data) > 0:
            self.state['window'] = deque(data[self.column].tail(self.period
                ).tolist(), maxlen=self.period)
            self.state['sum'] = sum(self.state['window'])
            self.is_initialized = True
            self.last_timestamp = data.index[-1]
        return result

    @with_exception_handling
    def update(self, new_data_point: Dict[str, Union[float, datetime]]) ->Dict[
        str, float]:
        """
        Update the SMA with a new data point.
        
        Args:
            new_data_point: Dictionary with a new OHLCV data point
            
        Returns:
            Dictionary with the updated SMA value
        """
        if not self.is_initialized:
            logger.error('SMA not initialized, cannot update')
            return {}
        try:
            new_price = new_data_point.get(self.column)
            if new_price is None:
                logger.error(
                    f'Column {self.column} not found in new data point')
                return {}
            window = self.state['window']
            current_sum = self.state['sum']
            if len(window) >= self.period:
                current_sum = current_sum - window[0] + new_price
                window.append(new_price)
            else:
                window.append(new_price)
                current_sum += new_price
            self.state['sum'] = current_sum
            sma_value = current_sum / len(window)
            if 'timestamp' in new_data_point:
                self.last_timestamp = new_data_point['timestamp']
            return {self.column_name: sma_value}
        except Exception as e:
            logger.error(f'Error updating SMA: {str(e)}')
            return {}

    def get_output_columns(self) ->List[str]:
        """
        Get the names of the output columns produced by this indicator.
        
        Returns:
            List of column names
        """
        return [self.column_name]


class IncrementalEMA(IncrementalIndicator):
    """
    Incremental Exponential Moving Average (EMA).
    
    This class implements an Exponential Moving Average that can be updated
    efficiently when new data arrives, maintaining only the previous EMA value
    and the smoothing factor.
    """

    def __init__(self, name: str='EMA', params: Dict[str, Any]=None):
        """
        Initialize the incremental EMA.
        
        Args:
            name: Base name for the EMA columns
            params: Dictionary of parameters for the EMA
                    - period: Period for the EMA (default: 14)
                    - column: Column to compute EMA for (default: "close")
        """
        params = params or {}
        self.period = params.get('period', 14)
        self.column = params.get('column', 'close')
        self.column_name = f'{name}_{self.period}'
        self.alpha = 2 / (self.period + 1)
        super().__init__(name, params)

    def initialize(self, data: pd.DataFrame) ->pd.DataFrame:
        """
        Initialize the EMA with historical data.
        
        Args:
            data: Historical OHLCV data
            
        Returns:
            DataFrame with EMA values added
        """
        if self.column not in data.columns:
            logger.error(f'Column {self.column} not found in data')
            self.is_initialized = False
            return data
        result = data.copy()
        result[self.column_name] = result[self.column].ewm(span=self.period,
            adjust=False).mean()
        if len(data) > 0:
            self.state['previous_ema'] = result[self.column_name].iloc[-1]
            self.is_initialized = True
            self.last_timestamp = data.index[-1]
        return result

    @with_exception_handling
    def update(self, new_data_point: Dict[str, Union[float, datetime]]) ->Dict[
        str, float]:
        """
        Update the EMA with a new data point.
        
        Args:
            new_data_point: Dictionary with a new OHLCV data point
            
        Returns:
            Dictionary with the updated EMA value
        """
        if not self.is_initialized:
            logger.error('EMA not initialized, cannot update')
            return {}
        try:
            new_price = new_data_point.get(self.column)
            if new_price is None:
                logger.error(
                    f'Column {self.column} not found in new data point')
                return {}
            previous_ema = self.state['previous_ema']
            ema_value = new_price * self.alpha + previous_ema * (1 - self.alpha
                )
            self.state['previous_ema'] = ema_value
            if 'timestamp' in new_data_point:
                self.last_timestamp = new_data_point['timestamp']
            return {self.column_name: ema_value}
        except Exception as e:
            logger.error(f'Error updating EMA: {str(e)}')
            return {}

    def get_output_columns(self) ->List[str]:
        """
        Get the names of the output columns produced by this indicator.
        
        Returns:
            List of column names
        """
        return [self.column_name]


class IncrementalMACD(IncrementalIndicator):
    """
    Incremental Moving Average Convergence Divergence (MACD).
    
    This class implements a MACD indicator that can be updated efficiently when new data
    arrives by using incremental EMAs for all its components.
    """

    def __init__(self, name: str='MACD', params: Dict[str, Any]=None):
        """
        Initialize the incremental MACD.
        
        Args:
            name: Name for the MACD columns
            params: Dictionary of parameters for the MACD:
                    - fast_period: Period for fast EMA (default: 12)
                    - slow_period: Period for slow EMA (default: 26)
                    - signal_period: Period for signal line EMA (default: 9)
                    - column: Price column to use (default: "close")
        """
        params = params or {}
        self.fast_period = params.get('fast_period', 12)
        self.slow_period = params.get('slow_period', 26)
        self.signal_period = params.get('signal_period', 9)
        self.column = params.get('column', 'close')
        self.macd_line_name = f'{name}_line'
        self.signal_line_name = f'{name}_signal'
        self.histogram_name = f'{name}_hist'
        self.fast_ema = IncrementalEMA(name='fast_ema', params={'period':
            self.fast_period, 'column': self.column})
        self.slow_ema = IncrementalEMA(name='slow_ema', params={'period':
            self.slow_period, 'column': self.column})
        self.signal_ema = IncrementalEMA(name='signal_ema', params={
            'period': self.signal_period, 'column': 'macd_line'})
        super().__init__(name, params)

    def initialize(self, data: pd.DataFrame) ->pd.DataFrame:
        """
        Initialize the MACD with historical data.
        
        Args:
            data: Historical OHLCV data
            
        Returns:
            DataFrame with MACD values added
        """
        result = data.copy()
        result = self.fast_ema.initialize(result)
        result = self.slow_ema.initialize(result)
        fast_col = self.fast_ema.get_output_columns()[0]
        slow_col = self.slow_ema.get_output_columns()[0]
        if fast_col not in result.columns or slow_col not in result.columns:
            logger.error('Failed to initialize EMA components for MACD')
            self.is_initialized = False
            return data
        result[self.macd_line_name] = result[fast_col] - result[slow_col]
        signal_data = pd.DataFrame({'close': result[self.macd_line_name]})
        signal_data.index = result.index
        signal_result = self.signal_ema.initialize(signal_data)
        signal_col = self.signal_ema.get_output_columns()[0]
        if signal_col in signal_result.columns:
            result[self.signal_line_name] = signal_result[signal_col]
            result[self.histogram_name] = result[self.macd_line_name] - result[
                self.signal_line_name]
        if len(data) > 0:
            self.state['previous_macd'] = result[self.macd_line_name].iloc[-1]
            self.is_initialized = True
            self.last_timestamp = data.index[-1]
        if 'fast_ema_' in result.columns:
            result = result.drop(columns=['fast_ema_' + str(self.fast_period)])
        if 'slow_ema_' in result.columns:
            result = result.drop(columns=['slow_ema_' + str(self.slow_period)])
        return result

    @with_exception_handling
    def update(self, new_data_point: Dict[str, Union[float, datetime]]) ->Dict[
        str, float]:
        """
        Update the MACD with a new data point.
        
        Args:
            new_data_point: Dictionary with a new OHLCV data point
            
        Returns:
            Dictionary with the updated MACD values
        """
        if not self.is_initialized:
            logger.error('MACD not initialized, cannot update')
            return {}
        try:
            fast_update = self.fast_ema.update(new_data_point)
            slow_update = self.slow_ema.update(new_data_point)
            if not fast_update or not slow_update:
                logger.error('Failed to update EMA components for MACD')
                return {}
            fast_val = fast_update[self.fast_ema.get_output_columns()[0]]
            slow_val = slow_update[self.slow_ema.get_output_columns()[0]]
            macd_line_value = fast_val - slow_val
            signal_data_point = {'close': macd_line_value, 'timestamp':
                new_data_point.get('timestamp')}
            signal_update = self.signal_ema.update(signal_data_point)
            if not signal_update:
                logger.error('Failed to update signal line for MACD')
                return {}
            signal_line_value = signal_update[self.signal_ema.
                get_output_columns()[0]]
            histogram_value = macd_line_value - signal_line_value
            self.state['previous_macd'] = macd_line_value
            if 'timestamp' in new_data_point:
                self.last_timestamp = new_data_point['timestamp']
            return {self.macd_line_name: macd_line_value, self.
                signal_line_name: signal_line_value, self.histogram_name:
                histogram_value}
        except Exception as e:
            logger.error(f'Error updating MACD: {str(e)}')
            return {}

    def get_output_columns(self) ->List[str]:
        """
        Get the names of the output columns produced by this indicator.
        
        Returns:
            List of column names
        """
        return [self.macd_line_name, self.signal_line_name, self.histogram_name
            ]
