"""
Seasonal Analysis Indicator.

This module implements seasonal decomposition with pattern detection
for day-of-week, monthly, and yearly seasonal effects,
along with dynamic strength calculation of seasonal patterns.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from enum import Enum
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from datetime import datetime, timedelta
from feature_store_service.indicators.base_indicator import BaseIndicator


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class SeasonalityType(Enum):
    """Enum for types of seasonality."""
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'
    QUARTERLY = 'quarterly'
    YEARLY = 'yearly'


class SeasonalAnalysisIndicator(BaseIndicator):
    """
    Seasonal Analysis Indicator.
    
    This indicator implements seasonal decomposition with pattern detection
    for day-of-week, monthly, and yearly seasonal effects,
    along with dynamic strength calculation of seasonal patterns.
    """
    category = 'seasonal'

    def __init__(self, timeframe: str='D', seasonal_periods: Dict[str, int]
        =None, decomposition_method: str='STL', seasonal_strength_window:
        int=252, column: str='close', **kwargs):
        """
        Initialize Seasonal Analysis indicator.
        
        Args:
            timeframe: Chart timeframe ('D' for daily, 'W' for weekly, etc.)
            seasonal_periods: Dictionary of seasonality types and their periods
                              (defaults based on timeframe if None)
            decomposition_method: Method to use for decomposition ('STL' or 'classic')
            seasonal_strength_window: Window to calculate seasonal strength
            column: Data column to use for calculations (default: 'close')
            **kwargs: Additional parameters
        """
        self.timeframe = timeframe
        self.decomposition_method = decomposition_method
        self.seasonal_strength_window = seasonal_strength_window
        self.column = column
        self.name = 'seasonal_analysis'
        if seasonal_periods is None:
            if timeframe == 'D':
                self.seasonal_periods = {SeasonalityType.WEEKLY: 5,
                    SeasonalityType.MONTHLY: 21, SeasonalityType.QUARTERLY:
                    63, SeasonalityType.YEARLY: 252}
            elif timeframe == 'W':
                self.seasonal_periods = {SeasonalityType.MONTHLY: 4,
                    SeasonalityType.QUARTERLY: 13, SeasonalityType.YEARLY: 52}
            elif timeframe == 'M':
                self.seasonal_periods = {SeasonalityType.QUARTERLY: 3,
                    SeasonalityType.YEARLY: 12}
            else:
                self.seasonal_periods = {SeasonalityType.DAILY: 24,
                    SeasonalityType.WEEKLY: 24 * 7}
        else:
            self.seasonal_periods = seasonal_periods

    def calculate(self, data: pd.DataFrame) ->pd.DataFrame:
        """
        Perform seasonal analysis on the given data.
        
        Args:
            data: DataFrame with OHLCV data and datetime index
            
        Returns:
            DataFrame with seasonal analysis results
        """
        if self.column not in data.columns:
            raise ValueError(f"Data must contain '{self.column}' column")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError('Data index must be a DatetimeIndex')
        result = data.copy()
        self._extract_time_features(result)
        for seasonality_type, period in self.seasonal_periods.items():
            if len(result) >= period * 2:
                self._perform_decomposition(result, seasonality_type, period)
        self._calculate_seasonal_strength(result)
        self._calculate_combined_seasonal_factor(result)
        return result

    def _extract_time_features(self, data: pd.DataFrame) ->None:
        """
        Extract time-based features from datetime index.
        
        Args:
            data: DataFrame with datetime index
        """
        data['day_of_week'] = data.index.dayofweek
        data['hour_of_day'] = data.index.hour
        data['day_of_month'] = data.index.day
        data['month_of_year'] = data.index.month
        data['quarter'] = data.index.quarter
        data['week_of_year'] = data.index.isocalendar().week

    @with_exception_handling
    def _perform_decomposition(self, data: pd.DataFrame, seasonality_type:
        SeasonalityType, period: int) ->None:
        """
        Perform seasonal decomposition.
        
        Args:
            data: DataFrame with time series data
            seasonality_type: Type of seasonality to analyze
            period: Seasonal period for decomposition
        """
        price_series = data[self.column].copy()
        price_series = price_series.fillna(method='ffill').fillna(method=
            'bfill')
        if len(price_series) < period * 2:
            return
        try:
            if self.decomposition_method == 'STL':
                decomposition = STL(price_series, period=period, robust=True
                    ).fit()
                seasonal = decomposition.seasonal
                trend = decomposition.trend
                residual = decomposition.resid
            else:
                decomposition = seasonal_decompose(price_series, model=
                    'additive', period=period, extrapolate_trend='freq')
                seasonal = decomposition.seasonal
                trend = decomposition.trend
                residual = decomposition.resid
            data[f'seasonal_{seasonality_type.value}'] = seasonal
            data[f'trend_{seasonality_type.value}'] = trend
            data[f'residual_{seasonality_type.value}'] = residual
            self._calculate_average_seasonal_pattern(data, seasonality_type,
                period)
        except Exception as e:
            print(
                f'Error in seasonal decomposition for {seasonality_type.value}: {e}'
                )
            data[f'seasonal_{seasonality_type.value}'] = 0
            data[f'trend_{seasonality_type.value}'] = price_series
            data[f'residual_{seasonality_type.value}'] = 0

    def _calculate_average_seasonal_pattern(self, data: pd.DataFrame,
        seasonality_type: SeasonalityType, period: int) ->None:
        """
        Calculate average seasonal pattern for the given seasonality type.
        
        Args:
            data: DataFrame with decomposition results
            seasonality_type: Type of seasonality
            period: Seasonal period
        """
        seasonal_col = f'seasonal_{seasonality_type.value}'
        if seasonality_type == SeasonalityType.WEEKLY:
            pattern_col = 'day_of_week'
            group_col = 'Day of Week'
        elif seasonality_type == SeasonalityType.DAILY:
            pattern_col = 'hour_of_day'
            group_col = 'Hour of Day'
        elif seasonality_type == SeasonalityType.MONTHLY:
            pattern_col = 'day_of_month'
            group_col = 'Day of Month'
        elif seasonality_type == SeasonalityType.QUARTERLY:
            if 'day_of_quarter' not in data.columns:
                data['day_of_quarter'] = (data.index.month - 1
                    ) % 3 * 30 + data.index.day
            pattern_col = 'day_of_quarter'
            group_col = 'Day of Quarter'
        elif seasonality_type == SeasonalityType.YEARLY:
            if self.timeframe in ['D', 'H']:
                pattern_col = 'week_of_year'
                group_col = 'Week of Year'
            else:
                pattern_col = 'month_of_year'
                group_col = 'Month of Year'
        if pattern_col in data.columns:
            pattern = data.groupby(pattern_col)[seasonal_col].mean()
            pattern_name = f'{seasonality_type.value}_pattern'
            data[pattern_name] = data[pattern_col].map(pattern)
            pattern_dict = {f'{group_col} {idx}': val for idx, val in
                pattern.items()}
            if not hasattr(self, 'seasonal_patterns'):
                self.seasonal_patterns = {}
            self.seasonal_patterns[seasonality_type] = pattern_dict

    def _calculate_seasonal_strength(self, data: pd.DataFrame) ->None:
        """
        Calculate the strength of seasonal patterns.
        
        Args:
            data: DataFrame with decomposition results
        """
        for seasonality_type in self.seasonal_periods.keys():
            seasonal_col = f'seasonal_{seasonality_type.value}'
            residual_col = f'residual_{seasonality_type.value}'
            strength_col = f'{seasonality_type.value}_strength'
            if seasonal_col in data.columns and residual_col in data.columns:
                var_seasonal = data[seasonal_col].rolling(self.
                    seasonal_strength_window).var()
                var_residual = data[residual_col].rolling(self.
                    seasonal_strength_window).var()
                total_var = var_seasonal + var_residual
                data[strength_col] = np.where(total_var > 0, var_seasonal /
                    total_var * 100, 0)
                n = self.seasonal_strength_window
                if n > 0:
                    f_stat = var_seasonal / var_residual.replace(0, np.nan)
                    data[f'{seasonality_type.value}_significance'] = 1 / (1 +
                        f_stat)
                else:
                    data[f'{seasonality_type.value}_significance'] = np.nan

    def _calculate_combined_seasonal_factor(self, data: pd.DataFrame) ->None:
        """
        Calculate combined seasonal factor across all analyzed periods.
        
        Args:
            data: DataFrame with decomposition results
        """
        seasonal_cols = [f'seasonal_{st.value}' for st in self.
            seasonal_periods.keys() if f'seasonal_{st.value}' in data.columns]
        if seasonal_cols:
            data['combined_seasonal_factor'] = data[seasonal_cols].sum(axis=1)
            price_mean = data[self.column].mean()
            if price_mean != 0:
                data['seasonal_percentage'] = data['combined_seasonal_factor'
                    ] / price_mean * 100
            else:
                data['seasonal_percentage'] = 0
            data['seasonal_direction'] = np.sign(data[
                'combined_seasonal_factor'])
            strength_cols = [f'{st.value}_strength' for st in self.
                seasonal_periods.keys() if f'{st.value}_strength' in data.
                columns]
            if strength_cols:
                weights = np.linspace(1, 0.5, len(strength_cols))
                weighted_strengths = np.zeros(len(data))
                for i, col in enumerate(strength_cols):
                    weighted_strengths += weights[i] * data[col].fillna(0)
                data['combined_seasonal_strength'] = weighted_strengths / sum(
                    weights)
                data['seasonal_signal'] = data['seasonal_direction'] * data[
                    'combined_seasonal_strength']

    def get_seasonal_forecast(self, data: pd.DataFrame, forecast_periods:
        int=30) ->pd.DataFrame:
        """
        Generate seasonal forecast based on identified patterns.
        
        Args:
            data: DataFrame with calculated seasonal components
            forecast_periods: Number of periods to forecast
            
        Returns:
            DataFrame with forecasted seasonal components
        """
        last_date = data.index[-1]
        if self.timeframe == 'D':
            forecast_dates = pd.date_range(start=last_date + timedelta(days
                =1), periods=forecast_periods, freq='B')
        elif self.timeframe == 'H':
            forecast_dates = pd.date_range(start=last_date + timedelta(
                hours=1), periods=forecast_periods, freq='H')
        elif self.timeframe == 'W':
            forecast_dates = pd.date_range(start=last_date + timedelta(
                weeks=1), periods=forecast_periods, freq='W')
        elif self.timeframe == 'M':
            forecast_dates = pd.date_range(start=last_date + timedelta(days
                =31), periods=forecast_periods, freq='M')
        else:
            forecast_dates = pd.date_range(start=last_date + timedelta(days
                =1), periods=forecast_periods, freq='D')
        forecast = pd.DataFrame(index=forecast_dates)
        forecast['day_of_week'] = forecast.index.dayofweek
        forecast['hour_of_day'] = forecast.index.hour
        forecast['day_of_month'] = forecast.index.day
        forecast['month_of_year'] = forecast.index.month
        forecast['quarter'] = forecast.index.quarter
        forecast['week_of_year'] = forecast.index.isocalendar().week
        forecast['day_of_quarter'] = (forecast.index.month - 1
            ) % 3 * 30 + forecast.index.day
        trend_cols = [f'trend_{st.value}' for st in self.seasonal_periods.
            keys() if f'trend_{st.value}' in data.columns]
        if trend_cols:
            trend_series = data[trend_cols[0]].dropna()
            x = np.arange(len(trend_series))
            y = trend_series.values
            a, b = np.polyfit(x, y, 1)
            x_forecast = np.arange(len(trend_series), len(trend_series) +
                len(forecast))
            forecast['trend'] = a * x_forecast + b
        for seasonality_type in self.seasonal_periods.keys():
            pattern_name = f'{seasonality_type.value}_pattern'
            if pattern_name in data.columns:
                if seasonality_type == SeasonalityType.WEEKLY:
                    pattern_col = 'day_of_week'
                elif seasonality_type == SeasonalityType.DAILY:
                    pattern_col = 'hour_of_day'
                elif seasonality_type == SeasonalityType.MONTHLY:
                    pattern_col = 'day_of_month'
                elif seasonality_type == SeasonalityType.QUARTERLY:
                    pattern_col = 'day_of_quarter'
                elif seasonality_type == SeasonalityType.YEARLY:
                    pattern_col = 'month_of_year' if self.timeframe in ['W',
                        'M'] else 'week_of_year'
                pattern_dict = data.groupby(pattern_col)[pattern_name].mean(
                    ).to_dict()
                forecast[f'seasonal_{seasonality_type.value}'] = forecast[
                    pattern_col].map(pattern_dict)
        seasonal_cols = [col for col in forecast.columns if col.startswith(
            'seasonal_')]
        if seasonal_cols:
            forecast['combined_seasonal_factor'] = forecast[seasonal_cols].sum(
                axis=1)
            if 'trend' in forecast.columns:
                forecast['seasonal_forecast'] = forecast['trend'] + forecast[
                    'combined_seasonal_factor']
        return forecast

    def get_seasonal_visualization_data(self, result: pd.DataFrame) ->Dict[
        str, Any]:
        """
        Get data for visualizing seasonal patterns.
        
        Args:
            result: DataFrame with calculated seasonal components
            
        Returns:
            Dictionary with visualization data
        """
        visualization_data = {'patterns': {}, 'strengths': {},
            'combined_seasonal_factor': result['combined_seasonal_factor'].
            tolist(), 'seasonal_signal': result['seasonal_signal'].tolist() if
            'seasonal_signal' in result.columns else []}
        for seasonality_type in self.seasonal_periods.keys():
            pattern_col = f'{seasonality_type.value}_pattern'
            strength_col = f'{seasonality_type.value}_strength'
            if pattern_col in result.columns:
                if seasonality_type == SeasonalityType.WEEKLY:
                    group_col = 'day_of_week'
                    labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                elif seasonality_type == SeasonalityType.DAILY:
                    group_col = 'hour_of_day'
                    labels = [f'{h}:00' for h in range(24)]
                elif seasonality_type == SeasonalityType.MONTHLY:
                    group_col = 'day_of_month'
                    labels = [str(d) for d in range(1, 32)]
                elif seasonality_type == SeasonalityType.QUARTERLY:
                    group_col = 'day_of_quarter'
                    labels = [str(d) for d in range(1, 92)]
                elif seasonality_type == SeasonalityType.YEARLY:
                    if self.timeframe in ['W', 'M']:
                        group_col = 'month_of_year'
                        labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    else:
                        group_col = 'week_of_year'
                        labels = [f'W{w}' for w in range(1, 54)]
                pattern = result.groupby(group_col)[pattern_col].mean(
                    ).reindex(range(len(labels))).fillna(0).tolist()
                visualization_data['patterns'][seasonality_type.value] = {
                    'labels': labels[:len(pattern)], 'values': pattern}
            if strength_col in result.columns:
                visualization_data['strengths'][seasonality_type.value
                    ] = result[strength_col].tolist()
        return visualization_data


""""""
