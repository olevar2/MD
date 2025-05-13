"""
VIX-based Volatility Module.

This module provides implementations of VIX-based volatility indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from core.base_indicator import BaseIndicator


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