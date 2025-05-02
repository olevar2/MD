"""
Volatility Indicators Module.

This module provides implementations of various volatility-based indicators.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from feature_store_service.indicators.base_indicator import BaseIndicator


class BollingerBands(BaseIndicator):
    """
    Bollinger Bands indicator.
    
    This volatility indicator creates bands around a moving average,
    with the width of the bands varying with volatility.
    """
    
    category = "volatility"
    
    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        column: str = "close",
        **kwargs
    ):
        """
        Initialize Bollinger Bands indicator.
        
        Args:
            window: Lookback period for the moving average
            num_std: Number of standard deviations for the bands
            column: Data column to use for calculations (default: 'close')
            **kwargs: Additional parameters
        """
        self.window = window
        self.num_std = num_std
        self.column = column
        self.name_middle = f"bb_middle_{window}"
        self.name_upper = f"bb_upper_{window}_{num_std}"
        self.name_lower = f"bb_lower_{window}_{num_std}"
        self.name_width = f"bb_width_{window}_{num_std}"
        self.name_pct_b = f"bb_pct_b_{window}_{num_std}"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Bollinger Bands values
        """
        if self.column not in data.columns:
            raise ValueError(f"Data must contain '{self.column}' column")
            
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate middle band (SMA)
        result[self.name_middle] = result[self.column].rolling(window=self.window).mean()
        
        # Calculate standard deviation
        rolling_std = result[self.column].rolling(window=self.window).std()
        
        # Calculate upper and lower bands
        result[self.name_upper] = result[self.name_middle] + (rolling_std * self.num_std)
        result[self.name_lower] = result[self.name_middle] - (rolling_std * self.num_std)
        
        # Calculate bandwidth
        result[self.name_width] = (result[self.name_upper] - result[self.name_lower]) / result[self.name_middle]
        
        # Calculate %B
        result[self.name_pct_b] = (result[self.column] - result[self.name_lower]) / (result[self.name_upper] - result[self.name_lower])
        
        return result


class AverageTrueRange(BaseIndicator):
    """
    Average True Range (ATR) indicator.
    
    This volatility indicator measures market volatility by decomposing the
    entire range of an asset price for a period.
    """
    
    category = "volatility"
    
    def __init__(self, window: int = 14, **kwargs):
        """
        Initialize Average True Range indicator.
        
        Args:
            window: Lookback period for the ATR calculation
            **kwargs: Additional parameters
        """
        self.window = window
        self.name = f"atr_{window}"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ATR for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with ATR values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate true range
        high_low = result['high'] - result['low']
        high_close_prev = abs(result['high'] - result['close'].shift(1))
        low_close_prev = abs(result['low'] - result['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate ATR using simple moving average
        # Note: For a more accurate Wilder's smoothing method, you would use a different approach
        result[self.name] = true_range.rolling(window=self.window).mean()
        
        return result


class KeltnerChannels(BaseIndicator):
    """
    Keltner Channels indicator.
    
    This volatility indicator uses ATR to set channel width, creating
    a dynamic envelope around a moving average.
    """
    
    category = "volatility"
    
    def __init__(
        self,
        window: int = 20,
        atr_window: int = 10,
        atr_multiplier: float = 2.0,
        ma_method: str = "ema",
        column: str = "close",
        **kwargs
    ):
        """
        Initialize Keltner Channels indicator.
        
        Args:
            window: Lookback period for the moving average
            atr_window: Lookback period for the ATR calculation
            atr_multiplier: Multiplier for the ATR
            ma_method: Moving average type ('sma' or 'ema')
            column: Data column to use for calculations (default: 'close')
            **kwargs: Additional parameters
        """
        self.window = window
        self.atr_window = atr_window
        self.atr_multiplier = atr_multiplier
        self.ma_method = ma_method
        self.column = column
        self.name_middle = f"kc_middle_{window}_{ma_method}"
        self.name_upper = f"kc_upper_{window}_{atr_window}_{atr_multiplier}"
        self.name_lower = f"kc_lower_{window}_{atr_window}_{atr_multiplier}"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Keltner Channels for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Keltner Channels values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate middle line
        if self.ma_method == 'sma':
            result[self.name_middle] = result[self.column].rolling(window=self.window).mean()
        elif self.ma_method == 'ema':
            result[self.name_middle] = result[self.column].ewm(span=self.window, adjust=False).mean()
        else:
            raise ValueError(f"Invalid ma_method: {self.ma_method}. Expected 'sma' or 'ema'.")
        
        # Calculate ATR
        high_low = result['high'] - result['low']
        high_close_prev = abs(result['high'] - result['close'].shift(1))
        low_close_prev = abs(result['low'] - result['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_window).mean()
        
        # Calculate upper and lower bands
        result[self.name_upper] = result[self.name_middle] + (atr * self.atr_multiplier)
        result[self.name_lower] = result[self.name_middle] - (atr * self.atr_multiplier)
        
        return result


class DonchianChannels(BaseIndicator):
    """
    Donchian Channels indicator with dynamic period optimization.
    
    This volatility indicator shows the highest high and lowest low over a specified period,
    creating channels that indicate price extremes and potential breakouts.
    """
    
    category = "volatility"
    
    def __init__(
        self,
        window: int = 20,
        optimize_period: bool = False,
        min_period: int = 10,
        max_period: int = 50,
        optimization_metric: str = "volatility_efficiency",
        **kwargs
    ):
        """
        Initialize Donchian Channels indicator.
        
        Args:
            window: Lookback period for the channel calculation
            optimize_period: Whether to dynamically optimize the period
            min_period: Minimum period to consider during optimization
            max_period: Maximum period to consider during optimization
            optimization_metric: Metric to use for optimization ('volatility_efficiency', 
                                'trend_strength', 'dynamic_range')
            **kwargs: Additional parameters
        """
        self.window = window
        self.optimize_period = optimize_period
        self.min_period = min_period
        self.max_period = max_period
        self.optimization_metric = optimization_metric
        self.optimized_window = None
        
        # Define output column names
        self.name_upper = f"donchian_upper_{window}"
        self.name_lower = f"donchian_lower_{window}"
        self.name_middle = f"donchian_middle_{window}"
        self.name_width = f"donchian_width_{window}"
        self.name_optimized_period = "donchian_optimized_period"
        
    def _optimize_period(self, data: pd.DataFrame) -> int:
        """
        Optimize the lookback period based on the selected metric.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Optimized lookback period
        """
        best_period = self.window
        best_score = -np.inf
        
        # Try different periods and evaluate each
        for period in range(self.min_period, self.max_period + 1):
            # Calculate upper and lower bands for this period
            high_max = data['high'].rolling(window=period).max()
            low_min = data['low'].rolling(window=period).min()
            middle = (high_max + low_min) / 2
            width = high_max - low_min
            
            # Calculate score based on the selected metric
            if self.optimization_metric == 'volatility_efficiency':
                # Higher score when channel width captures price movements efficiently
                recent_volatility = data['close'].pct_change().rolling(period).std()
                score = (width / (data['close'] * recent_volatility)).mean()
            
            elif self.optimization_metric == 'trend_strength':
                # Higher score when price stays near the extremes (trending market)
                distance_from_middle = abs(data['close'] - middle)
                score = (distance_from_middle / (width / 2)).mean()
            
            elif self.optimization_metric == 'dynamic_range':
                # Higher score when channel adapts to changing volatility
                volatility_change = width.pct_change().rolling(period).std()
                score = 1 / volatility_change.mean()
            
            else:
                raise ValueError(f"Invalid optimization_metric: {self.optimization_metric}")
                
            # Update best period if we found a better score
            if not np.isnan(score) and score > best_score:
                best_score = score
                best_period = period
                
        return best_period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Donchian Channels for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Donchian Channels values
        """
        required_cols = ['high', 'low']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Optimize period if requested
        if self.optimize_period and len(data) >= self.max_period * 3:  # Need enough data for optimization
            self.optimized_window = self._optimize_period(data)
            window = self.optimized_window
            result[self.name_optimized_period] = self.optimized_window
        else:
            window = self.window
            
        # Calculate channel components
        result[self.name_upper] = result['high'].rolling(window=window).max()
        result[self.name_lower] = result['low'].rolling(window=window).min()
        result[self.name_middle] = (result[self.name_upper] + result[self.name_lower]) / 2
        result[self.name_width] = result[self.name_upper] - result[self.name_lower]
        
        return result


class PriceEnvelopes(BaseIndicator):
    """
    Price Envelopes indicator with customizable percentage options.
    
    This volatility indicator creates upper and lower bands by adding/subtracting a 
    percentage of the price from a moving average.
    """
    
    category = "volatility"
    
    def __init__(
        self,
        window: int = 20,
        percent: float = 2.5,
        ma_method: str = "sma",
        column: str = "close",
        adaptive_percentage: bool = False,
        **kwargs
    ):
        """
        Initialize Price Envelopes indicator.
        
        Args:
            window: Lookback period for the moving average
            percent: Percentage for the envelopes (e.g., 2.5 for 2.5%)
            ma_method: Moving average type ('sma', 'ema', or 'wma')
            column: Data column to use for calculations
            adaptive_percentage: Whether to adapt the percentage based on volatility
            **kwargs: Additional parameters
        """
        self.window = window
        self.percent = percent
        self.ma_method = ma_method.lower()
        self.column = column
        self.adaptive_percentage = adaptive_percentage
        
        # Define output column names
        self.name_ma = f"envelope_{ma_method}_{window}"
        self.name_upper = f"envelope_upper_{window}_{percent}"
        self.name_lower = f"envelope_lower_{window}_{percent}"
        self.name_adaptive = "envelope_adaptive_percent" if adaptive_percentage else None
        
    def _calculate_ma(self, data: pd.Series, window: int, method: str) -> pd.Series:
        """
        Calculate moving average based on the specified method.
        
        Args:
            data: Price series
            window: Lookback period
            method: Moving average type ('sma', 'ema', or 'wma')
            
        Returns:
            Series with moving average values
        """
        if method == 'sma':
            return data.rolling(window=window).mean()
        elif method == 'ema':
            return data.ewm(span=window, adjust=False).mean()
        elif method == 'wma':
            weights = np.arange(1, window + 1)
            return data.rolling(window=window).apply(
                lambda x: np.sum(weights * x) / np.sum(weights), raw=True
            )
        else:
            raise ValueError(f"Invalid ma_method: {method}. Expected 'sma', 'ema', or 'wma'.")
            
    def _calculate_adaptive_percentage(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate adaptive percentage based on recent volatility.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with adaptive percentage values
        """
        # Calculate price volatility using standard deviation of returns
        volatility = data[self.column].pct_change().rolling(window=self.window).std()
        
        # Scale the base percentage by the ratio of current volatility to average volatility
        mean_volatility = volatility.mean()
        if mean_volatility > 0:
            adaptive_pct = self.percent * (volatility / mean_volatility)
            # Limit the range to avoid extreme values
            return adaptive_pct.clip(lower=self.percent * 0.5, upper=self.percent * 2.0)
        else:
            return pd.Series(self.percent, index=data.index)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Price Envelopes for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Price Envelopes values
        """
        if self.column not in data.columns:
            raise ValueError(f"Data must contain '{self.column}' column")
            
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate moving average
        result[self.name_ma] = self._calculate_ma(result[self.column], self.window, self.ma_method)
        
        # Calculate adaptive percentage if enabled
        if self.adaptive_percentage:
            percent_series = self._calculate_adaptive_percentage(data)
            result[self.name_adaptive] = percent_series
        else:
            percent_series = pd.Series(self.percent, index=data.index)
        
        # Calculate upper and lower envelopes
        envelope_factor = percent_series / 100
        result[self.name_upper] = result[self.name_ma] * (1 + envelope_factor)
        result[self.name_lower] = result[self.name_ma] * (1 - envelope_factor)
        
        return result


class VIXFixIndicator(BaseIndicator):
    """
    VIX Fix Indicator with support for various volatility metrics.
    
    This indicator adapts the CBOE Volatility Index (VIX) methodology to individual 
    assets, measuring implied volatility using price ranges rather than options data.
    """
    
    category = "volatility"
    
    def __init__(
        self,
        period: int = 22,
        atr_period: int = 10,
        std_dev_period: int = 22,
        normalization_period: int = 100,
        metric_type: str = "close_to_close",
        **kwargs
    ):
        """
        Initialize VIX Fix Indicator.
        
        Args:
            period: Lookback period for the main calculation
            atr_period: Period for ATR calculation when using range-based metrics
            std_dev_period: Period for standard deviation calculation
            normalization_period: Period for normalizing to historical volatility
            metric_type: Type of volatility metric to use ('close_to_close', 
                        'parkinson', 'garman_klass', 'rogers_satchell')
            **kwargs: Additional parameters
        """
        self.period = period
        self.atr_period = atr_period
        self.std_dev_period = std_dev_period
        self.normalization_period = normalization_period
        self.metric_type = metric_type.lower()
        
        # Define output column names
        self.name = f"vix_fix_{period}_{metric_type}"
        self.name_normalized = f"vix_fix_norm_{period}_{metric_type}"
        self.name_signal = f"vix_fix_signal_{period}_{metric_type}"
        
    def _calculate_parkinson_volatility(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """
        Calculate Parkinson volatility based on high-low range.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            
        Returns:
            Series with Parkinson volatility values
        """
        # Parkinson volatility formula uses log of high/low ratio
        log_hl_ratio = np.log(high / low)
        return np.sqrt((1.0 / (4.0 * np.log(2.0))) * (log_hl_ratio ** 2))
    
    def _calculate_garman_klass_volatility(
        self, open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """
        Calculate Garman-Klass volatility.
        
        Args:
            open_: Series of open prices
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            
        Returns:
            Series with Garman-Klass volatility values
        """
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)
        return np.sqrt(0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2))
    
    def _calculate_rogers_satchell_volatility(
        self, open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """
        Calculate Rogers-Satchell volatility.
        
        Args:
            open_: Series of open prices
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            
        Returns:
            Series with Rogers-Satchell volatility values
        """
        log_ho = np.log(high / open_)
        log_hc = np.log(high / close)
        log_lo = np.log(low / open_)
        log_lc = np.log(low / close)
        return np.sqrt(log_ho * log_hc + log_lo * log_lc)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VIX Fix Indicator for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with VIX Fix Indicator values
        """
        required_cols = ['close']
        if self.metric_type in ['parkinson', 'garman_klass', 'rogers_satchell']:
            required_cols.extend(['open', 'high', 'low'])
            
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate raw volatility based on the selected metric
        if self.metric_type == 'close_to_close':
            # Traditional close-to-close volatility (returns standard deviation)
            log_returns = np.log(result['close'] / result['close'].shift(1))
            raw_volatility = log_returns.rolling(window=self.std_dev_period).std() * np.sqrt(252)
            
        elif self.metric_type == 'parkinson':
            # Parkinson volatility uses high-low range
            daily_vol = self._calculate_parkinson_volatility(result['high'], result['low'])
            raw_volatility = daily_vol.rolling(window=self.period).mean() * np.sqrt(252)
            
        elif self.metric_type == 'garman_klass':
            # Garman-Klass volatility uses OHLC data
            daily_vol = self._calculate_garman_klass_volatility(
                result['open'], result['high'], result['low'], result['close']
            )
            raw_volatility = daily_vol.rolling(window=self.period).mean() * np.sqrt(252)
            
        elif self.metric_type == 'rogers_satchell':
            # Rogers-Satchell volatility uses OHLC data
            daily_vol = self._calculate_rogers_satchell_volatility(
                result['open'], result['high'], result['low'], result['close']
            )
            raw_volatility = daily_vol.rolling(window=self.period).mean() * np.sqrt(252)
            
        else:
            raise ValueError(f"Invalid metric_type: {self.metric_type}")
        
        # Apply VIX Fix transformation (percentage change from minimum volatility)
        min_vol = raw_volatility.rolling(window=self.period).min()
        result[self.name] = 100 * ((raw_volatility / min_vol) - 1.0)
        
        # Calculate normalized version (percentile relative to history)
        if len(result) >= self.normalization_period:
            result[self.name_normalized] = result[self.name].rolling(
                window=self.normalization_period
            ).apply(
                lambda x: pd.Series(x).rank().iloc[-1] / len(x) * 100
            )
            
            # Generate signals based on extreme values
            result[self.name_signal] = 0
            result.loc[result[self.name_normalized] > 80, self.name_signal] = -1  # Extremely high volatility
            result.loc[result[self.name_normalized] < 20, self.name_signal] = 1   # Extremely low volatility
        
        return result


class HistoricalVolatility(BaseIndicator):
    """
    Historical Volatility indicator with multiple variants.
    
    This indicator calculates volatility using various methodologies including
    historical, implied (approximated), and projected volatility metrics.
    """
    
    category = "volatility"
    
    def __init__(
        self,
        period: int = 21,
        variant: str = "historical",
        projection_periods: int = 5,
        scaling: str = "annualized",
        smoothing_period: int = 5,
        confidence_level: float = 0.95,
        **kwargs
    ):
        """
        Initialize Historical Volatility indicator.
        
        Args:
            period: Lookback period for the volatility calculation
            variant: Type of volatility to calculate ('historical', 'implied', 'projected', 'garch')
            projection_periods: Number of periods to project volatility forward
            scaling: Scaling factor for volatility ('annualized', 'daily', 'weekly', 'monthly')
            smoothing_period: Period for smoothing the raw volatility values
            confidence_level: Confidence level for VaR and volatility cone calculations
            **kwargs: Additional parameters
        """
        self.period = period
        self.variant = variant.lower()
        self.projection_periods = projection_periods
        self.scaling = scaling.lower()
        self.smoothing_period = smoothing_period
        self.confidence_level = confidence_level
        
        # Define output column names
        self.name = f"hist_vol_{self.variant}_{period}"
        self.name_smooth = f"hist_vol_{self.variant}_{period}_smooth_{smoothing_period}"
        self.name_projected = f"hist_vol_projected_{projection_periods}"
        self.name_upper_band = f"hist_vol_upper_band_{period}"
        self.name_lower_band = f"hist_vol_lower_band_{period}"
        self.name_var = f"hist_vol_var_{period}_{int(confidence_level * 100)}"
        
    def _get_scaling_factor(self) -> float:
        """
        Get the appropriate scaling factor based on the selected scaling.
        
        Returns:
            Scaling factor for volatility calculations
        """
        if self.scaling == "annualized":
            return np.sqrt(252)  # Trading days in a year
        elif self.scaling == "weekly":
            return np.sqrt(5)    # Trading days in a week
        elif self.scaling == "monthly":
            return np.sqrt(21)   # Trading days in a month
        else:  # daily
            return 1.0
    
    def _calculate_garch_volatility(self, returns: pd.Series) -> pd.Series:
        """
        Calculate volatility using a simple GARCH(1,1) model.
        
        Args:
            returns: Series of price returns
            
        Returns:
            Series with GARCH volatility estimates
        """
        try:
            import arch
            from arch import arch_model
        except ImportError:
            raise ImportError("GARCH calculations require the 'arch' package. Install with 'pip install arch'.")
        
        # Filter out NaN values
        clean_returns = returns.dropna()
        
        if len(clean_returns) < self.period * 2:
            return pd.Series(np.nan, index=returns.index)
        
        # Fit GARCH(1,1) model
        model = arch_model(clean_returns, vol='Garch', p=1, q=1)
        try:
            model_fit = model.fit(disp='off')
            
            # Generate conditional volatility forecast
            forecasts = model_fit.forecast(horizon=self.projection_periods)
            conditional_vol = np.sqrt(forecasts.variance.iloc[-1])
            
            # Create a series with the last value being the forecast
            garch_vol = pd.Series(np.nan, index=returns.index)
            garch_vol.iloc[-1] = conditional_vol[0] * self._get_scaling_factor()
            
            return garch_vol
        except:
            # Fall back to historical volatility if GARCH fitting fails
            return returns.rolling(window=self.period).std() * self._get_scaling_factor()
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Historical Volatility for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Historical Volatility values
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
            
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate log returns
        log_returns = np.log(result['close'] / result['close'].shift(1))
        
        # Get scaling factor
        scaling_factor = self._get_scaling_factor()
        
        # Calculate base historical volatility
        hist_vol = log_returns.rolling(window=self.period).std() * scaling_factor
        
        if self.variant == 'historical':
            # Standard historical volatility
            result[self.name] = hist_vol
            
        elif self.variant == 'implied':
            # Approximate implied volatility using historical and recent price movements
            # Note: True implied volatility requires options data
            recent_vol = log_returns.rolling(window=int(self.period/3)).std() * scaling_factor
            result[self.name] = hist_vol * 0.7 + recent_vol * 0.3
            
        elif self.variant == 'projected':
            # Project volatility forward using simple autoregressive model
            result[self.name] = hist_vol
            
            # Calculate projected volatility using simple exponential weighting of past volatilities
            weights = np.exp(-np.arange(self.projection_periods) / (self.projection_periods / 2))
            weights = weights / weights.sum()
            
            vol_history = hist_vol.rolling(window=self.projection_periods).apply(
                lambda x: np.sum(weights * x[::-1]), raw=True
            )
            result[self.name_projected] = vol_history
            
        elif self.variant == 'garch':
            # GARCH-based volatility
            result[self.name] = self._calculate_garch_volatility(log_returns)
            
        else:
            raise ValueError(f"Invalid variant: {self.variant}")
        
        # Calculate smoothed volatility
        result[self.name_smooth] = result[self.name].rolling(window=self.smoothing_period).mean()
        
        # Calculate volatility bands (volatility of volatility)
        vol_std = result[self.name].rolling(window=self.period).std()
        result[self.name_upper_band] = result[self.name] + vol_std * 2
        result[self.name_lower_band] = result[self.name].clip(lower=0) - vol_std * 2
        
        # Calculate Value at Risk (VaR) based on volatility
        z_score = {0.9: 1.282, 0.95: 1.645, 0.99: 2.326}.get(self.confidence_level, 1.645)
        result[self.name_var] = result['close'] * z_score * result[self.name] / scaling_factor
        
        return result
