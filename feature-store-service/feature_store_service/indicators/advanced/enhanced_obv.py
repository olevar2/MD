"""
Advanced On-Balance Volume Indicator.

This module implements On-Balance Volume with multilevel analysis,
regression channels, and volume trend strength metrics.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from enum import Enum
from sklearn.linear_model import LinearRegression
from scipy import stats

from feature_store_service.indicators.base_indicator import BaseIndicator


class EnhancedOBVIndicator(BaseIndicator):
    """
    Enhanced On-Balance Volume Indicator.
    
    This indicator implements On-Balance Volume (OBV) with multilevel analysis,
    regression channels, and volume trend strength metrics.
    """
    
    category = "volume"
    
    def __init__(
        self, 
        short_period: int = 20,
        medium_period: int = 50,
        long_period: int = 100,
        regression_period: int = 20,
        channel_width: float = 2.0,
        column_close: str = "close",
        column_volume: str = "volume",
        **kwargs
    ):
        """
        Initialize Enhanced On-Balance Volume indicator.
        
        Args:
            short_period: Short-term smoothing period for OBV
            medium_period: Medium-term smoothing period for OBV
            long_period: Long-term smoothing period for OBV
            regression_period: Period for regression channel calculations
            channel_width: Width of regression channels in standard deviations
            column_close: Column name for closing prices
            column_volume: Column name for volume data
            **kwargs: Additional parameters
        """
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period
        self.regression_period = regression_period
        self.channel_width = channel_width
        self.column_close = column_close
        self.column_volume = column_volume
        self.name = "enhanced_obv"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate enhanced OBV indicator on the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with enhanced OBV indicator values
        """
        # Check required columns
        required_columns = [self.column_close, self.column_volume]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Data is missing required columns: {', '.join(missing_columns)}")
            
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate basic OBV
        self._calculate_basic_obv(result)
        
        # Calculate smoothed OBV
        self._calculate_smoothed_obv(result)
        
        # Calculate OBV regression channels
        self._calculate_regression_channels(result)
        
        # Calculate OBV trend metrics
        self._calculate_trend_metrics(result)
        
        return result
    
    def _calculate_basic_obv(self, data: pd.DataFrame) -> None:
        """
        Calculate basic On-Balance Volume.
        
        Args:
            data: DataFrame to update with OBV
        """
        # Get close and volume data
        close = data[self.column_close]
        volume = data[self.column_volume]
        
        # Handle missing volume data
        volume = volume.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        # Calculate price changes
        price_change = close.diff()
        
        # Initialize OBV array
        obv = np.zeros(len(data))
        
        # First OBV value is just the first volume value
        if len(volume) > 0:
            obv[0] = volume.iloc[0]
        
        # Calculate OBV
        for i in range(1, len(data)):
            if price_change.iloc[i] > 0:
                # Close higher than previous close -> add volume
                obv[i] = obv[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:
                # Close lower than previous close -> subtract volume
                obv[i] = obv[i-1] - volume.iloc[i]
            else:
                # Close equal to previous close -> no change
                obv[i] = obv[i-1]
        
        # Add OBV to dataframe
        data['obv'] = obv
    
    def _calculate_smoothed_obv(self, data: pd.DataFrame) -> None:
        """
        Calculate smoothed OBV at different timeframes.
        
        Args:
            data: DataFrame to update with smoothed OBV
        """
        # Calculate EMA of OBV at different timeframes
        data['obv_short'] = data['obv'].ewm(span=self.short_period, adjust=False).mean()
        data['obv_medium'] = data['obv'].ewm(span=self.medium_period, adjust=False).mean()
        data['obv_long'] = data['obv'].ewm(span=self.long_period, adjust=False).mean()
        
        # Calculate rate of change for OBV
        data['obv_roc'] = data['obv'].pct_change(10) * 100
        
        # Calculate OBV divergence from price
        # (OBV percent change minus price percent change)
        data['obv_price_divergence'] = (
            data['obv'].pct_change(20) - 
            data[self.column_close].pct_change(20)
        ) * 100
    
    def _calculate_regression_channels(self, data: pd.DataFrame) -> None:
        """
        Calculate regression channels for OBV.
        
        Args:
            data: DataFrame to update with regression channels
        """
        # Pre-allocate columns
        data['obv_regression_mid'] = np.nan
        data['obv_regression_upper'] = np.nan
        data['obv_regression_lower'] = np.nan
        data['obv_channel_position'] = np.nan
        
        # Minimum data points needed
        if len(data) < self.regression_period:
            return
        
        # Apply rolling regression to create channels
        for i in range(self.regression_period, len(data) + 1):
            # Get OBV data for current window
            y = data.iloc[i - self.regression_period:i]['obv'].values
            X = np.arange(self.regression_period).reshape(-1, 1)
            
            # Skip windows with NaN values
            if np.isnan(y).any():
                continue
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate regression line for the last point
            regression_value = model.predict(np.array([[self.regression_period - 1]]))[0]
            
            # Calculate standard deviation of residuals
            y_pred = model.predict(X)
            residuals = y - y_pred
            residual_std = np.std(residuals)
            
            # Calculate channel boundaries
            upper_channel = regression_value + self.channel_width * residual_std
            lower_channel = regression_value - self.channel_width * residual_std
            
            # Store in DataFrame at the current position (end of window)
            current_idx = i - 1
            data.iloc[current_idx, data.columns.get_loc('obv_regression_mid')] = regression_value
            data.iloc[current_idx, data.columns.get_loc('obv_regression_upper')] = upper_channel
            data.iloc[current_idx, data.columns.get_loc('obv_regression_lower')] = lower_channel
            
            # Calculate channel position (0-100%)
            current_obv = data.iloc[current_idx]['obv']
            if upper_channel > lower_channel:  # Prevent division by zero
                channel_position = (current_obv - lower_channel) / (upper_channel - lower_channel) * 100
                data.iloc[current_idx, data.columns.get_loc('obv_channel_position')] = channel_position
        
        # Calculate channel breakout signals
        data['obv_channel_breakout'] = np.where(
            data['obv'] > data['obv_regression_upper'],
            1,  # Upper breakout
            np.where(
                data['obv'] < data['obv_regression_lower'],
                -1,  # Lower breakout
                0  # Inside channel
            )
        )
    
    def _calculate_trend_metrics(self, data: pd.DataFrame) -> None:
        """
        Calculate trend metrics based on OBV.
        
        Args:
            data: DataFrame to update with trend metrics
        """
        # Calculate slopes for trend strength
        for period, name in [
            (self.short_period, 'short'), 
            (self.medium_period, 'medium'), 
            (self.long_period, 'long')
        ]:
            if len(data) >= period:
                data[f'obv_{name}_slope'] = self._calculate_rolling_slope(
                    data['obv'], period
                )
                
        # Calculate OBV trend strength based on slopes
        if all(col in data.columns for col in ['obv_short_slope', 'obv_medium_slope', 'obv_long_slope']):
            # Normalize slopes
            short_weight, medium_weight, long_weight = 0.5, 0.3, 0.2
            
            # Calculate weighted average of slopes
            data['obv_trend_strength'] = (
                data['obv_short_slope'] * short_weight +
                data['obv_medium_slope'] * medium_weight +
                data['obv_long_slope'] * long_weight
            )
            
            # Convert to percentage scale (-100 to +100)
            max_strength = data['obv_trend_strength'].abs().rolling(
                window=252, min_periods=20
            ).quantile(0.95).fillna(method='bfill').fillna(1.0)
            
            data['obv_trend_strength'] = (
                data['obv_trend_strength'] / max_strength * 100
            ).clip(-100, 100)
            
        # Calculate OBV trend direction
        data['obv_trend_direction'] = np.sign(data['obv_medium'].diff(5))
        
        # Calculate OBV trend consistency
        # (how often the short-term trend matches the medium-term trend)
        if 'obv_short_slope' in data.columns and 'obv_medium_slope' in data.columns:
            rolling_window = min(self.medium_period, 20)
            
            # Calculate sign agreement between short and medium slopes
            sign_match = (
                np.sign(data['obv_short_slope']) == np.sign(data['obv_medium_slope'])
            ).astype(int)
            
            # Calculate consistency as percentage of agreement in rolling window
            data['obv_trend_consistency'] = sign_match.rolling(
                window=rolling_window, min_periods=rolling_window // 2
            ).mean() * 100
    
    def _calculate_rolling_slope(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate the rolling slope of a time series.
        
        Args:
            series: Time series data
            period: Rolling window size
            
        Returns:
            Series with rolling slope values
        """
        slopes = pd.Series(np.nan, index=series.index)
        
        for i in range(period, len(series) + 1):
            # Get window of data
            y = series.iloc[i - period:i].values
            
            # Skip if any NaN values
            if np.isnan(y).any():
                continue
                
            # Create X values (0 to period-1)
            x = np.arange(period)
            
            # Calculate linear regression slope using Pearson correlation
            if len(y) >= 2:  # Need at least 2 points for correlation
                correlation, _ = stats.pearsonr(x, y)
                std_x = np.std(x)
                std_y = np.std(y)
                
                if std_x > 0 and std_y > 0:
                    slope = correlation * (std_y / std_x)
                    slopes.iloc[i - 1] = slope
        
        return slopes

    def get_obv_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive OBV analysis data.
        
        Args:
            data: DataFrame with calculated OBV data
            
        Returns:
            Dictionary with OBV analysis data
        """
        # Get most recent values
        last_row = data.iloc[-1]
        
        # Extract key metrics
        obv_trend_direction = int(last_row.get('obv_trend_direction', 0))
        obv_trend_strength = float(last_row.get('obv_trend_strength', 0))
        obv_trend_consistency = float(last_row.get('obv_trend_consistency', 0))
        
        # Determine channel position
        channel_position = float(last_row.get('obv_channel_position', 50))
        channel_breakout = int(last_row.get('obv_channel_breakout', 0))
        
        # Calculate divergence with price
        price_close = data[self.column_close].iloc[-1]
        price_change_pct = data[self.column_close].pct_change(20).iloc[-1] * 100
        obv_change_pct = data['obv'].pct_change(20).iloc[-1] * 100
        
        divergence = {
            'has_divergence': False,
            'type': 'none',
            'strength': 0.0
        }
        
        # Check for divergence (OBV and price moving in opposite directions)
        if abs(price_change_pct) > 0.5 and abs(obv_change_pct) > 0.5:
            price_direction = np.sign(price_change_pct)
            obv_direction = np.sign(obv_change_pct)
            
            if price_direction != obv_direction:
                divergence['has_divergence'] = True
                divergence_strength = min(abs(price_change_pct), abs(obv_change_pct))
                
                if price_direction > 0 and obv_direction < 0:
                    # Price up, OBV down: negative divergence (bearish)
                    divergence['type'] = 'negative'
                    divergence['strength'] = divergence_strength
                else:
                    # Price down, OBV up: positive divergence (bullish)
                    divergence['type'] = 'positive'
                    divergence['strength'] = divergence_strength
        
        # Return analysis
        return {
            'trend': {
                'direction': obv_trend_direction,
                'strength': obv_trend_strength,
                'consistency': obv_trend_consistency
            },
            'channel': {
                'position': channel_position,
                'breakout': channel_breakout
            },
            'divergence': divergence
        }
"""
