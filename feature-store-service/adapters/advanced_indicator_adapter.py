"""
Advanced Indicator Adapter Module

This module provides adapter implementations for advanced indicator interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import logging
import importlib
from datetime import datetime
from common_lib.indicators.indicator_interfaces import IBaseIndicator, IAdvancedIndicator, IFibonacciAnalyzer, IndicatorCategory, IIndicatorAdapter
logger = logging.getLogger(__name__)


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class AdvancedIndicatorAdapter(IIndicatorAdapter):
    """
    Adapter for advanced indicators that implements the common interface.

    This adapter can either wrap an actual indicator instance or provide
    standalone functionality to avoid circular dependencies.
    """

    @with_exception_handling
    def __init__(self, indicator_class=None, name_prefix='', **kwargs):
        """
        Initialize the adapter.

        Args:
            indicator_class: Optional indicator class to wrap
            name_prefix: Optional prefix for column names
            **kwargs: Additional parameters to pass to the indicator constructor
        """
        self.indicator_class = indicator_class
        self.indicator_instance = None
        self.name_prefix = name_prefix
        self.kwargs = kwargs
        if self.indicator_class:
            try:
                self.indicator_instance = self.indicator_class(**kwargs)
            except Exception as e:
                logger.warning(f'Error initializing indicator: {str(e)}')

    def adapt(self, source_indicator: Any) ->IBaseIndicator:
        """
        Adapt a source indicator to the IBaseIndicator interface.

        Args:
            source_indicator: Source indicator instance

        Returns:
            Adapted indicator implementing IBaseIndicator
        """
        self.indicator_instance = source_indicator
        return self

    @property
    def name(self) ->str:
        """Get the name of the indicator."""
        if hasattr(self.indicator_instance, 'name'):
            return f'{self.name_prefix}_{self.indicator_instance.name}'
        elif hasattr(self.indicator_instance, '__class__'):
            return (
                f'{self.name_prefix}_{self.indicator_instance.__class__.__name__}'
                )
        else:
            return f'{self.name_prefix}_unknown'

    @property
    @with_exception_handling
    def category(self) ->IndicatorCategory:
        """Get the category of the indicator."""
        if hasattr(self.indicator_instance, 'category'):
            try:
                return IndicatorCategory(self.indicator_instance.category)
            except ValueError:
                pass
        return IndicatorCategory.CUSTOM

    @property
    def params(self) ->Dict[str, Any]:
        """Get the parameters for the indicator."""
        if hasattr(self.indicator_instance, 'params'):
            return self.indicator_instance.params
        else:
            return self.kwargs

    @with_exception_handling
    def calculate(self, data: pd.DataFrame) ->pd.DataFrame:
        """
        Calculate the indicator values.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with indicator values added as columns
        """
        if self.indicator_instance:
            try:
                if hasattr(self.indicator_instance, 'calculate'):
                    return self.indicator_instance.calculate(data)
                elif callable(self.indicator_instance):
                    return self.indicator_instance(data)
            except Exception as e:
                logger.warning(f'Error calculating indicator: {str(e)}')
        return data

    @with_exception_handling
    def get_column_names(self) ->List[str]:
        """
        Get the names of columns added by this indicator.

        Returns:
            List of column names
        """
        if hasattr(self.indicator_instance, 'get_column_names'):
            return self.indicator_instance.get_column_names()
        elif hasattr(self.indicator_instance, 'output_names'):
            return self.indicator_instance.output_names
        else:
            try:
                sample_data = pd.DataFrame({'open': np.random.random(100),
                    'high': np.random.random(100), 'low': np.random.random(
                    100), 'close': np.random.random(100), 'volume': np.
                    random.random(100)})
                before_cols = set(sample_data.columns)
                result = self.calculate(sample_data)
                after_cols = set(result.columns)
                return list(after_cols - before_cols)
            except:
                return []


class FibonacciAnalyzerAdapter(IFibonacciAnalyzer):
    """
    Adapter for Fibonacci analysis components that implements the common interface.

    This adapter can either wrap an actual analyzer instance or provide
    standalone functionality to avoid circular dependencies.
    """

    def __init__(self, analyzer_instance=None):
        """
        Initialize the adapter.

        Args:
            analyzer_instance: Optional actual analyzer instance to wrap
        """
        self.analyzer = analyzer_instance

    @with_exception_handling
    def calculate_retracements(self, data: pd.DataFrame, high_col: str=
        'high', low_col: str='low', levels: List[float]=None) ->pd.DataFrame:
        """
        Calculate Fibonacci retracement levels.

        Args:
            data: DataFrame with OHLCV data
            high_col: Name of the high price column
            low_col: Name of the low price column
            levels: Optional list of Fibonacci levels

        Returns:
            DataFrame with retracement levels added
        """
        if self.analyzer and hasattr(self.analyzer, 'calculate_retracements'):
            try:
                return self.analyzer.calculate_retracements(data=data,
                    high_col=high_col, low_col=low_col, levels=levels)
            except Exception as e:
                logger.warning(f'Error calculating retracements: {str(e)}')
        if levels is None:
            levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        result = data.copy()
        high_price = data[high_col].max()
        low_price = data[low_col].min()
        range_price = high_price - low_price
        for level in levels:
            level_str = str(level).replace('.', '_')
            result[f'fib_retracement_{level_str}'
                ] = high_price - range_price * level
        return result

    @with_exception_handling
    def calculate_extensions(self, data: pd.DataFrame, high_col: str='high',
        low_col: str='low', close_col: str='close', levels: List[float]=None
        ) ->pd.DataFrame:
        """
        Calculate Fibonacci extension levels.

        Args:
            data: DataFrame with OHLCV data
            high_col: Name of the high price column
            low_col: Name of the low price column
            close_col: Name of the close price column
            levels: Optional list of Fibonacci levels

        Returns:
            DataFrame with extension levels added
        """
        if self.analyzer and hasattr(self.analyzer, 'calculate_extensions'):
            try:
                return self.analyzer.calculate_extensions(data=data,
                    high_col=high_col, low_col=low_col, close_col=close_col,
                    levels=levels)
            except Exception as e:
                logger.warning(f'Error calculating extensions: {str(e)}')
        if levels is None:
            levels = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618]
        result = data.copy()
        high_price = data[high_col].max()
        low_price = data[low_col].min()
        range_price = high_price - low_price
        for level in levels:
            level_str = str(level).replace('.', '_')
            result[f'fib_extension_{level_str}'
                ] = high_price + range_price * level
        return result

    @with_exception_handling
    def calculate_arcs(self, data: pd.DataFrame, high_col: str='high',
        low_col: str='low', levels: List[float]=None) ->pd.DataFrame:
        """
        Calculate Fibonacci arcs.

        Args:
            data: DataFrame with OHLCV data
            high_col: Name of the high price column
            low_col: Name of the low price column
            levels: Optional list of Fibonacci levels

        Returns:
            DataFrame with arc levels added
        """
        if self.analyzer and hasattr(self.analyzer, 'calculate_arcs'):
            try:
                return self.analyzer.calculate_arcs(data=data, high_col=
                    high_col, low_col=low_col, levels=levels)
            except Exception as e:
                logger.warning(f'Error calculating arcs: {str(e)}')
        if levels is None:
            levels = [0.382, 0.5, 0.618]
        result = data.copy()
        high_price = data[high_col].max()
        low_price = data[low_col].min()
        range_price = high_price - low_price
        for level in levels:
            level_str = str(level).replace('.', '_')
            result[f'fib_arc_{level_str}'] = high_price - range_price * level
        return result

    @with_exception_handling
    def calculate_fans(self, data: pd.DataFrame, high_col: str='high',
        low_col: str='low', levels: List[float]=None) ->pd.DataFrame:
        """
        Calculate Fibonacci fans.

        Args:
            data: DataFrame with OHLCV data
            high_col: Name of the high price column
            low_col: Name of the low price column
            levels: Optional list of Fibonacci levels

        Returns:
            DataFrame with fan levels added
        """
        if self.analyzer and hasattr(self.analyzer, 'calculate_fans'):
            try:
                return self.analyzer.calculate_fans(data=data, high_col=
                    high_col, low_col=low_col, levels=levels)
            except Exception as e:
                logger.warning(f'Error calculating fans: {str(e)}')
        if levels is None:
            levels = [0.382, 0.5, 0.618]
        result = data.copy()
        high_price = data[high_col].max()
        low_price = data[low_col].min()
        range_price = high_price - low_price
        for level in levels:
            level_str = str(level).replace('.', '_')
            result[f'fib_fan_{level_str}'] = high_price - range_price * level
        return result

    @with_exception_handling
    def calculate_time_zones(self, data: pd.DataFrame, pivot_idx: int=None,
        levels: List[int]=None) ->pd.DataFrame:
        """
        Calculate Fibonacci time zones.

        Args:
            data: DataFrame with OHLCV data
            pivot_idx: Optional pivot index
            levels: Optional list of Fibonacci levels

        Returns:
            DataFrame with time zone levels added
        """
        if self.analyzer and hasattr(self.analyzer, 'calculate_time_zones'):
            try:
                return self.analyzer.calculate_time_zones(data=data,
                    pivot_idx=pivot_idx, levels=levels)
            except Exception as e:
                logger.warning(f'Error calculating time zones: {str(e)}')
        if levels is None:
            levels = [1, 2, 3, 5, 8, 13, 21, 34]
        result = data.copy()
        if pivot_idx is None:
            pivot_idx = 0
        for level in levels:
            zone_idx = pivot_idx + level
            if zone_idx < len(data):
                result.loc[result.index[zone_idx], f'fib_time_zone_{level}'
                    ] = 1
        return result


@with_exception_handling
def load_advanced_indicators() ->Dict[str, Any]:
    """
    Load advanced indicators from analysis engine if available.

    Returns:
        Dictionary of indicator classes
    """
    indicators = {}
    try:
        analysis_engine = importlib.import_module('analysis_engine')
        try:
            advanced_ta = importlib.import_module(
                'analysis_engine.analysis.advanced_ta')
            for attr_name in dir(advanced_ta):
                if attr_name.startswith('_'):
                    continue
                try:
                    attr = getattr(advanced_ta, attr_name)
                    if isinstance(attr, type) and hasattr(attr, 'calculate'):
                        indicators[attr_name] = attr
                except Exception as e:
                    logger.debug(
                        f'Error loading indicator {attr_name}: {str(e)}')
        except ImportError:
            logger.debug('Advanced TA module not available')
        try:
            pattern_recognition = importlib.import_module(
                'analysis_engine.analysis.pattern_recognition')
            for attr_name in dir(pattern_recognition):
                if attr_name.startswith('_'):
                    continue
                try:
                    attr = getattr(pattern_recognition, attr_name)
                    if isinstance(attr, type) and (hasattr(attr,
                        'find_patterns') or hasattr(attr, 'recognize') or
                        hasattr(attr, 'calculate')):
                        indicators[attr_name] = attr
                except Exception as e:
                    logger.debug(
                        f'Error loading pattern recognizer {attr_name}: {str(e)}'
                        )
        except ImportError:
            logger.debug('Pattern recognition module not available')
    except ImportError:
        logger.debug('Analysis engine module not available')
    return indicators
