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


class SeasonalityType(Enum):
    """Enum for types of seasonality."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class SeasonalAnalysisIndicator(BaseIndicator):
    """
    Seasonal Analysis Indicator.
    
    This indicator implements seasonal decomposition with pattern detection
    for day-of-week, monthly, and yearly seasonal effects,
    along with dynamic strength calculation of seasonal patterns.
    """
    
    category = "seasonal"
    
    def __init__(
        self, 
        timeframe: str = "D",
        seasonal_periods: Dict[str, int] = None,
        decomposition_method: str = "STL",
        seasonal_strength_window: int = 252,
        column: str = "close",
        **kwargs
    ):
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
        self.name = "seasonal_analysis"
        
        # Set default seasonal periods based on timeframe if not provided
        if seasonal_periods is None:
            if timeframe == 'D':  # Daily data
                self.seasonal_periods = {
                    SeasonalityType.WEEKLY: 5,  # 5 trading days in a week
                    SeasonalityType.MONTHLY: 21,  # ~21 trading days in a month
                    SeasonalityType.QUARTERLY: 63,  # ~63 trading days in a quarter
                    SeasonalityType.YEARLY: 252,  # ~252 trading days in a year
                }
            elif timeframe == 'W':  # Weekly data
                self.seasonal_periods = {
                    SeasonalityType.MONTHLY: 4,  # 4 weeks in a month
                    SeasonalityType.QUARTERLY: 13,  # 13 weeks in a quarter
                    SeasonalityType.YEARLY: 52,  # 52 weeks in a year
                }
            elif timeframe == 'M':  # Monthly data
                self.seasonal_periods = {
                    SeasonalityType.QUARTERLY: 3,  # 3 months in a quarter
                    SeasonalityType.YEARLY: 12,  # 12 months in a year
                }
            else:  # Default (H or lower timeframes)
                self.seasonal_periods = {
                    SeasonalityType.DAILY: 24,  # 24 hours in a day
                    SeasonalityType.WEEKLY: 24 * 7,  # Hours in a week
                }
        else:
            self.seasonal_periods = seasonal_periods
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
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
            raise ValueError("Data index must be a DatetimeIndex")
            
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Extract time features
        self._extract_time_features(result)
        
        # Perform decomposition for each seasonal period
        for seasonality_type, period in self.seasonal_periods.items():
            if len(result) >= period * 2:  # Need at least 2 full periods
                self._perform_decomposition(result, seasonality_type, period)
                
        # Calculate seasonal strength and significance
        self._calculate_seasonal_strength(result)
        
        # Calculate combined seasonal factor
        self._calculate_combined_seasonal_factor(result)
        
        return result
    
    def _extract_time_features(self, data: pd.DataFrame) -> None:
        """
        Extract time-based features from datetime index.
        
        Args:
            data: DataFrame with datetime index
        """
        # Extract day of week (0=Monday, 6=Sunday)
        data['day_of_week'] = data.index.dayofweek
        
        # Extract hour of day
        data['hour_of_day'] = data.index.hour
        
        # Extract day of month
        data['day_of_month'] = data.index.day
        
        # Extract month of year
        data['month_of_year'] = data.index.month
        
        # Extract quarter
        data['quarter'] = data.index.quarter
        
        # Extract week of year
        data['week_of_year'] = data.index.isocalendar().week
    
    def _perform_decomposition(self, data: pd.DataFrame, seasonality_type: SeasonalityType, period: int) -> None:
        """
        Perform seasonal decomposition.
        
        Args:
            data: DataFrame with time series data
            seasonality_type: Type of seasonality to analyze
            period: Seasonal period for decomposition
        """
        # Get the price series
        price_series = data[self.column].copy()
        
        # Fill any NaN values using forward fill then backward fill
        price_series = price_series.fillna(method='ffill').fillna(method='bfill')
        
        if len(price_series) < period * 2:
            return  # Not enough data for this period
            
        try:
            # Perform decomposition
            if self.decomposition_method == "STL":
                # STL decomposition (handles non-linear trends better)
                decomposition = STL(
                    price_series, 
                    period=period, 
                    robust=True
                ).fit()
                
                seasonal = decomposition.seasonal
                trend = decomposition.trend
                residual = decomposition.resid
                
            else:
                # Classical decomposition
                decomposition = seasonal_decompose(
                    price_series, 
                    model='additive', 
                    period=period, 
                    extrapolate_trend='freq'
                )
                
                seasonal = decomposition.seasonal
                trend = decomposition.trend
                residual = decomposition.resid
            
            # Add decomposition results to the dataframe
            data[f"seasonal_{seasonality_type.value}"] = seasonal
            data[f"trend_{seasonality_type.value}"] = trend
            data[f"residual_{seasonality_type.value}"] = residual
            
            # Calculate average seasonal patterns
            self._calculate_average_seasonal_pattern(data, seasonality_type, period)
                
        except Exception as e:
            print(f"Error in seasonal decomposition for {seasonality_type.value}: {e}")
            # Fill with zeros if decomposition fails
            data[f"seasonal_{seasonality_type.value}"] = 0
            data[f"trend_{seasonality_type.value}"] = price_series
            data[f"residual_{seasonality_type.value}"] = 0
    
    def _calculate_average_seasonal_pattern(self, data: pd.DataFrame, seasonality_type: SeasonalityType, period: int) -> None:
        """
        Calculate average seasonal pattern for the given seasonality type.
        
        Args:
            data: DataFrame with decomposition results
            seasonality_type: Type of seasonality
            period: Seasonal period
        """
        seasonal_col = f"seasonal_{seasonality_type.value}"
        
        if seasonality_type == SeasonalityType.WEEKLY:
            # Group by day of week
            pattern_col = 'day_of_week'
            group_col = 'Day of Week'
            
        elif seasonality_type == SeasonalityType.DAILY:
            # Group by hour of day
            pattern_col = 'hour_of_day'
            group_col = 'Hour of Day'
            
        elif seasonality_type == SeasonalityType.MONTHLY:
            # Group by day of month
            pattern_col = 'day_of_month'
            group_col = 'Day of Month'
            
        elif seasonality_type == SeasonalityType.QUARTERLY:
            # Group by day within quarter
            # First calculate day of quarter
            if 'day_of_quarter' not in data.columns:
                data['day_of_quarter'] = ((data.index.month - 1) % 3) * 30 + data.index.day
            pattern_col = 'day_of_quarter'
            group_col = 'Day of Quarter'
            
        elif seasonality_type == SeasonalityType.YEARLY:
            # Group by day or month of year
            if self.timeframe in ['D', 'H']:
                # For daily or hourly data, use week of year
                pattern_col = 'week_of_year'
                group_col = 'Week of Year'
            else:
                # For weekly/monthly data, use month of year
                pattern_col = 'month_of_year'
                group_col = 'Month of Year'
        
        # Calculate average seasonal pattern
        if pattern_col in data.columns:
            pattern = data.groupby(pattern_col)[seasonal_col].mean()
            
            # Create columns for the pattern values
            pattern_name = f"{seasonality_type.value}_pattern"
            
            # Map the pattern to each row
            data[pattern_name] = data[pattern_col].map(pattern)
            
            # Create a dictionary to store the pattern
            pattern_dict = {f"{group_col} {idx}": val for idx, val in pattern.items()}
            
            # Store the pattern dictionary as a property
            if not hasattr(self, 'seasonal_patterns'):
                self.seasonal_patterns = {}
            self.seasonal_patterns[seasonality_type] = pattern_dict
    
    def _calculate_seasonal_strength(self, data: pd.DataFrame) -> None:
        """
        Calculate the strength of seasonal patterns.
        
        Args:
            data: DataFrame with decomposition results
        """
        # Initialize seasonal strength columns
        for seasonality_type in self.seasonal_periods.keys():
            seasonal_col = f"seasonal_{seasonality_type.value}"
            residual_col = f"residual_{seasonality_type.value}"
            strength_col = f"{seasonality_type.value}_strength"
            
            if seasonal_col in data.columns and residual_col in data.columns:
                # Calculate F-statistic-like measure of seasonal strength
                # Variance of seasonal component / (Variance of seasonal + Variance of residual)
                var_seasonal = data[seasonal_col].rolling(self.seasonal_strength_window).var()
                var_residual = data[residual_col].rolling(self.seasonal_strength_window).var()
                
                # Avoid division by zero
                total_var = var_seasonal + var_residual
                data[strength_col] = np.where(
                    total_var > 0,
                    (var_seasonal / total_var) * 100,  # Convert to percentage
                    0
                )
                
                # Calculate p-value-like measure of seasonal significance
                # Lower values indicate more significant seasonality
                n = self.seasonal_strength_window
                if n > 0:
                    f_stat = var_seasonal / var_residual.replace(0, np.nan)
                    data[f"{seasonality_type.value}_significance"] = 1 / (1 + f_stat)
                else:
                    data[f"{seasonality_type.value}_significance"] = np.nan
    
    def _calculate_combined_seasonal_factor(self, data: pd.DataFrame) -> None:
        """
        Calculate combined seasonal factor across all analyzed periods.
        
        Args:
            data: DataFrame with decomposition results
        """
        # Get seasonal columns
        seasonal_cols = [f"seasonal_{st.value}" for st in self.seasonal_periods.keys()
                         if f"seasonal_{st.value}" in data.columns]
        
        if seasonal_cols:
            # Sum all seasonal factors
            data["combined_seasonal_factor"] = data[seasonal_cols].sum(axis=1)
            
            # Normalize to percentage of price
            price_mean = data[self.column].mean()
            if price_mean != 0:
                data["seasonal_percentage"] = (data["combined_seasonal_factor"] / price_mean) * 100
            else:
                data["seasonal_percentage"] = 0
                
            # Calculate if current combined seasonal factor is bullish or bearish
            data["seasonal_direction"] = np.sign(data["combined_seasonal_factor"])
            
            # Calculate combined seasonal strength
            strength_cols = [f"{st.value}_strength" for st in self.seasonal_periods.keys()
                             if f"{st.value}_strength" in data.columns]
            
            if strength_cols:
                # Take weighted average of all seasonal strengths
                # Give higher weight to shorter-term seasonality
                weights = np.linspace(1, 0.5, len(strength_cols))
                weighted_strengths = np.zeros(len(data))
                
                for i, col in enumerate(strength_cols):
                    weighted_strengths += weights[i] * data[col].fillna(0)
                    
                data["combined_seasonal_strength"] = weighted_strengths / sum(weights)
                
                # Calculate overall seasonal signal (-100 to +100)
                # Combine direction and strength
                data["seasonal_signal"] = data["seasonal_direction"] * data["combined_seasonal_strength"]

    def get_seasonal_forecast(self, data: pd.DataFrame, forecast_periods: int = 30) -> pd.DataFrame:
        """
        Generate seasonal forecast based on identified patterns.
        
        Args:
            data: DataFrame with calculated seasonal components
            forecast_periods: Number of periods to forecast
            
        Returns:
            DataFrame with forecasted seasonal components
        """
        # Create a forecast dataframe
        last_date = data.index[-1]
        
        # Create date range for forecast
        if self.timeframe == 'D':
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                          periods=forecast_periods, freq='B')  # Business days
        elif self.timeframe == 'H':
            forecast_dates = pd.date_range(start=last_date + timedelta(hours=1), 
                                          periods=forecast_periods, freq='H')
        elif self.timeframe == 'W':
            forecast_dates = pd.date_range(start=last_date + timedelta(weeks=1), 
                                          periods=forecast_periods, freq='W')
        elif self.timeframe == 'M':
            forecast_dates = pd.date_range(start=last_date + timedelta(days=31), 
                                          periods=forecast_periods, freq='M')
        else:
            # Default to daily
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                          periods=forecast_periods, freq='D')
            
        forecast = pd.DataFrame(index=forecast_dates)
        
        # Extract time features for the forecast dates
        forecast['day_of_week'] = forecast.index.dayofweek
        forecast['hour_of_day'] = forecast.index.hour
        forecast['day_of_month'] = forecast.index.day
        forecast['month_of_year'] = forecast.index.month
        forecast['quarter'] = forecast.index.quarter
        forecast['week_of_year'] = forecast.index.isocalendar().week
        forecast['day_of_quarter'] = ((forecast.index.month - 1) % 3) * 30 + forecast.index.day
        
        # Forecast trend component using simple linear extrapolation
        trend_cols = [f"trend_{st.value}" for st in self.seasonal_periods.keys()
                     if f"trend_{st.value}" in data.columns]
        
        # Use the longest-term trend available for forecasting
        if trend_cols:
            trend_series = data[trend_cols[0]].dropna()
            
            # Fit linear model to the trend
            x = np.arange(len(trend_series))
            y = trend_series.values
            a, b = np.polyfit(x, y, 1)
            
            # Extrapolate trend
            x_forecast = np.arange(len(trend_series), len(trend_series) + len(forecast))
            forecast['trend'] = a * x_forecast + b
        
        # Forecast seasonal components using the calculated patterns
        for seasonality_type in self.seasonal_periods.keys():
            pattern_name = f"{seasonality_type.value}_pattern"
            
            if pattern_name in data.columns:
                # Get the appropriate time feature for this seasonality
                if seasonality_type == SeasonalityType.WEEKLY:
                    pattern_col = 'day_of_week'
                elif seasonality_type == SeasonalityType.DAILY:
                    pattern_col = 'hour_of_day'
                elif seasonality_type == SeasonalityType.MONTHLY:
                    pattern_col = 'day_of_month'
                elif seasonality_type == SeasonalityType.QUARTERLY:
                    pattern_col = 'day_of_quarter'
                elif seasonality_type == SeasonalityType.YEARLY:
                    pattern_col = 'month_of_year' if self.timeframe in ['W', 'M'] else 'week_of_year'
                
                # Get the pattern for each date in the forecast
                pattern_dict = data.groupby(pattern_col)[pattern_name].mean().to_dict()
                
                # Map pattern to forecast
                forecast[f"seasonal_{seasonality_type.value}"] = forecast[pattern_col].map(pattern_dict)
        
        # Calculate combined seasonal forecast
        seasonal_cols = [col for col in forecast.columns if col.startswith('seasonal_')]
        if seasonal_cols:
            forecast['combined_seasonal_factor'] = forecast[seasonal_cols].sum(axis=1)
            
            # Add trend to get total forecast
            if 'trend' in forecast.columns:
                forecast['seasonal_forecast'] = forecast['trend'] + forecast['combined_seasonal_factor']
        
        return forecast
    
    def get_seasonal_visualization_data(self, result: pd.DataFrame) -> Dict[str, Any]:
        """
        Get data for visualizing seasonal patterns.
        
        Args:
            result: DataFrame with calculated seasonal components
            
        Returns:
            Dictionary with visualization data
        """
        visualization_data = {
            "patterns": {},
            "strengths": {},
            "combined_seasonal_factor": result["combined_seasonal_factor"].tolist(),
            "seasonal_signal": result["seasonal_signal"].tolist() if "seasonal_signal" in result.columns else []
        }
        
        # Add pattern data
        for seasonality_type in self.seasonal_periods.keys():
            pattern_col = f"{seasonality_type.value}_pattern"
            strength_col = f"{seasonality_type.value}_strength"
            
            if pattern_col in result.columns:
                # Find the right grouping column
                if seasonality_type == SeasonalityType.WEEKLY:
                    group_col = 'day_of_week'
                    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                elif seasonality_type == SeasonalityType.DAILY:
                    group_col = 'hour_of_day'
                    labels = [f"{h}:00" for h in range(24)]
                elif seasonality_type == SeasonalityType.MONTHLY:
                    group_col = 'day_of_month'
                    labels = [str(d) for d in range(1, 32)]
                elif seasonality_type == SeasonalityType.QUARTERLY:
                    group_col = 'day_of_quarter'
                    labels = [str(d) for d in range(1, 92)]
                elif seasonality_type == SeasonalityType.YEARLY:
                    if self.timeframe in ['W', 'M']:
                        group_col = 'month_of_year'
                        labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    else:
                        group_col = 'week_of_year'
                        labels = [f"W{w}" for w in range(1, 54)]
                
                # Get the pattern
                pattern = result.groupby(group_col)[pattern_col].mean().reindex(
                    range(len(labels))).fillna(0).tolist()
                
                visualization_data["patterns"][seasonality_type.value] = {
                    "labels": labels[:len(pattern)],
                    "values": pattern
                }
            
            if strength_col in result.columns:
                visualization_data["strengths"][seasonality_type.value] = result[strength_col].tolist()
        
        return visualization_data
""""""
