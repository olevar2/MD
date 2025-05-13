"""
Incremental Updater

This module provides utilities for incremental updates of data and calculations.
"""
import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, TypeVar, Generic
logger = logging.getLogger(__name__)
T = TypeVar('T')
U = TypeVar('U')


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class IncrementalUpdater(Generic[T, U]):
    """
    Generic incremental updater for data and calculations.
    
    This class provides a framework for incrementally updating data and calculations
    without recalculating everything from scratch.
    
    Type parameters:
        T: Type of the data
        U: Type of the result
    """

    def __init__(self, calculate_func: Callable[[T], U], update_func:
        Callable[[U, T], U], max_incremental_updates: int=10):
        """
        Initialize the incremental updater.
        
        Args:
            calculate_func: Function to calculate the result from scratch
            update_func: Function to update the result incrementally
            max_incremental_updates: Maximum number of incremental updates before recalculating
        """
        self.calculate_func = calculate_func
        self.update_func = update_func
        self.max_incremental_updates = max_incremental_updates
        self.result = None
        self.last_data = None
        self.update_count = 0

    def update(self, data: T) ->U:
        """
        Update the result.
        
        Args:
            data: New data
            
        Returns:
            Updated result
        """
        if self.result is None or self.last_data is None:
            self.result = self.calculate_func(data)
            self.last_data = data
            self.update_count = 0
            logger.debug('Initial calculation')
            return self.result
        if self.update_count >= self.max_incremental_updates:
            self.result = self.calculate_func(data)
            self.last_data = data
            self.update_count = 0
            logger.debug('Recalculation after max incremental updates')
            return self.result
        self.result = self.update_func(self.result, data)
        self.last_data = data
        self.update_count += 1
        logger.debug(f'Incremental update {self.update_count}')
        return self.result

    def force_recalculate(self, data: T) ->U:
        """
        Force recalculation from scratch.
        
        Args:
            data: Data
            
        Returns:
            Recalculated result
        """
        self.result = self.calculate_func(data)
        self.last_data = data
        self.update_count = 0
        logger.debug('Forced recalculation')
        return self.result

    @with_resilience('get_result')
    def get_result(self) ->Optional[U]:
        """
        Get the current result.
        
        Returns:
            Current result
        """
        return self.result


class DataFrameUpdater:
    """
    Incremental updater for pandas DataFrames.
    
    This class provides utilities for incrementally updating pandas DataFrames
    and calculations based on them.
    """

    def __init__(self, calculate_func: Callable[[pd.DataFrame], Any],
        max_incremental_updates: int=10, timestamp_column: str='timestamp'):
        """
        Initialize the DataFrame updater.
        
        Args:
            calculate_func: Function to calculate the result from scratch
            max_incremental_updates: Maximum number of incremental updates before recalculating
            timestamp_column: Name of the timestamp column
        """
        self.calculate_func = calculate_func
        self.max_incremental_updates = max_incremental_updates
        self.timestamp_column = timestamp_column
        self.result = None
        self.last_data = None
        self.update_count = 0

    def update(self, data: pd.DataFrame) ->Any:
        """
        Update the result.
        
        Args:
            data: New data
            
        Returns:
            Updated result
        """
        if self.result is None or self.last_data is None:
            self.result = self.calculate_func(data)
            self.last_data = data.copy()
            self.update_count = 0
            logger.debug('Initial calculation')
            return self.result
        if self.update_count >= self.max_incremental_updates:
            self.result = self.calculate_func(data)
            self.last_data = data.copy()
            self.update_count = 0
            logger.debug('Recalculation after max incremental updates')
            return self.result
        new_rows = self._get_new_rows(data)
        if new_rows.empty:
            logger.debug('No new rows')
            return self.result
        self.result = self._update_incrementally(new_rows)
        self.last_data = pd.concat([self.last_data, new_rows]).drop_duplicates(
            subset=[self.timestamp_column])
        self.update_count += 1
        logger.debug(
            f'Incremental update {self.update_count} with {len(new_rows)} new rows'
            )
        return self.result

    def _get_new_rows(self, data: pd.DataFrame) ->pd.DataFrame:
        """
        Get new rows from the data.
        
        Args:
            data: New data
            
        Returns:
            New rows
        """
        if self.timestamp_column not in data.columns:
            logger.warning(
                f'Timestamp column {self.timestamp_column} not found in data')
            return pd.DataFrame()
        last_timestamp = self.last_data[self.timestamp_column].max()
        new_rows = data[data[self.timestamp_column] > last_timestamp].copy()
        return new_rows

    def _update_incrementally(self, new_rows: pd.DataFrame) ->Any:
        """
        Update the result incrementally.
        
        Args:
            new_rows: New rows
            
        Returns:
            Updated result
        """
        return self.calculate_func(pd.concat([self.last_data, new_rows]))

    def force_recalculate(self, data: pd.DataFrame) ->Any:
        """
        Force recalculation from scratch.
        
        Args:
            data: Data
            
        Returns:
            Recalculated result
        """
        self.result = self.calculate_func(data)
        self.last_data = data.copy()
        self.update_count = 0
        logger.debug('Forced recalculation')
        return self.result

    @with_resilience('get_result')
    def get_result(self) ->Any:
        """
        Get the current result.
        
        Returns:
            Current result
        """
        return self.result


class TechnicalIndicatorUpdater(DataFrameUpdater):
    """
    Incremental updater for technical indicators.
    
    This class provides utilities for incrementally updating technical indicators
    without recalculating everything from scratch.
    """

    def __init__(self, indicator_func: Callable[[pd.DataFrame], pd.Series],
        max_incremental_updates: int=10, timestamp_column: str='timestamp',
        price_column: str='close', window: int=14):
        """
        Initialize the technical indicator updater.
        
        Args:
            indicator_func: Function to calculate the indicator from scratch
            max_incremental_updates: Maximum number of incremental updates before recalculating
            timestamp_column: Name of the timestamp column
            price_column: Name of the price column
            window: Window size for the indicator
        """
        super().__init__(calculate_func=indicator_func,
            max_incremental_updates=max_incremental_updates,
            timestamp_column=timestamp_column)
        self.price_column = price_column
        self.window = window

    def _update_incrementally(self, new_rows: pd.DataFrame) ->pd.Series:
        """
        Update the indicator incrementally.
        
        Args:
            new_rows: New rows
            
        Returns:
            Updated indicator
        """
        last_values = self.last_data.tail(self.window)[self.price_column
            ].values
        new_values = new_rows[self.price_column].values
        combined_values = np.concatenate([last_values, new_values])
        temp_df = pd.DataFrame({self.price_column: combined_values})
        indicator = self.calculate_func(temp_df)
        new_indicator = indicator.tail(len(new_values))
        return pd.concat([self.result, new_indicator])


class MovingAverageUpdater(TechnicalIndicatorUpdater):
    """
    Incremental updater for moving averages.
    
    This class provides a more efficient implementation for incrementally
    updating moving averages.
    """

    def _update_incrementally(self, new_rows: pd.DataFrame) ->pd.Series:
        """
        Update the moving average incrementally.
        
        Args:
            new_rows: New rows
            
        Returns:
            Updated moving average
        """
        last_values = self.last_data.tail(self.window)[self.price_column
            ].values
        new_values = new_rows[self.price_column].values
        last_ma = self.result.iloc[-1]
        new_ma_values = []
        for i, new_value in enumerate(new_values):
            oldest_value = last_values[i]
            last_values = np.append(last_values[1:], new_value)
            new_ma = last_ma + (new_value - oldest_value) / self.window
            new_ma_values.append(new_ma)
            last_ma = new_ma
        new_ma = pd.Series(new_ma_values, index=new_rows.index)
        return pd.concat([self.result, new_ma])


class RSIUpdater(TechnicalIndicatorUpdater):
    """
    Incremental updater for RSI.
    
    This class provides a more efficient implementation for incrementally
    updating RSI.
    """

    def __init__(self, indicator_func: Callable[[pd.DataFrame], pd.Series],
        max_incremental_updates: int=10, timestamp_column: str='timestamp',
        price_column: str='close', window: int=14):
        """
        Initialize the RSI updater.
        
        Args:
            indicator_func: Function to calculate RSI from scratch
            max_incremental_updates: Maximum number of incremental updates before recalculating
            timestamp_column: Name of the timestamp column
            price_column: Name of the price column
            window: Window size for RSI
        """
        super().__init__(indicator_func=indicator_func,
            max_incremental_updates=max_incremental_updates,
            timestamp_column=timestamp_column, price_column=price_column,
            window=window)
        self.last_avg_gain = None
        self.last_avg_loss = None

    def _update_incrementally(self, new_rows: pd.DataFrame) ->pd.Series:
        """
        Update RSI incrementally.
        
        Args:
            new_rows: New rows
            
        Returns:
            Updated RSI
        """
        if self.last_avg_gain is None or self.last_avg_loss is None:
            return super()._update_incrementally(new_rows)
        last_price = self.last_data[self.price_column].iloc[-1]
        new_values = new_rows[self.price_column].values
        new_rsi_values = []
        for new_price in new_values:
            change = new_price - last_price
            gain = max(0, change)
            loss = max(0, -change)
            self.last_avg_gain = (self.last_avg_gain * (self.window - 1) + gain
                ) / self.window
            self.last_avg_loss = (self.last_avg_loss * (self.window - 1) + loss
                ) / self.window
            rs = (self.last_avg_gain / self.last_avg_loss if self.
                last_avg_loss != 0 else float('inf'))
            rsi = 100 - 100 / (1 + rs)
            new_rsi_values.append(rsi)
            last_price = new_price
        new_rsi = pd.Series(new_rsi_values, index=new_rows.index)
        return pd.concat([self.result, new_rsi])

    def force_recalculate(self, data: pd.DataFrame) ->pd.Series:
        """
        Force recalculation of RSI from scratch.
        
        Args:
            data: Data
            
        Returns:
            Recalculated RSI
        """
        result = super().force_recalculate(data)
        self._update_state(data)
        return result

    def _update_state(self, data: pd.DataFrame) ->None:
        """
        Update the internal state for RSI calculation.
        
        Args:
            data: Data
        """
        changes = data[self.price_column].diff().dropna()
        gains = changes.copy()
        gains[gains < 0] = 0
        losses = -changes.copy()
        losses[losses < 0] = 0
        avg_gain = gains.rolling(window=self.window).mean().iloc[-1]
        avg_loss = losses.rolling(window=self.window).mean().iloc[-1]
        self.last_avg_gain = avg_gain
        self.last_avg_loss = avg_loss
