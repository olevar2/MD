"""
Historical Volatility Module.

This module provides implementations of historical volatility indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from feature_store_service.indicators.base_indicator import BaseIndicator


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