"""
Advanced Volatility Indicators Module

This module provides implementations of advanced volatility indicators including
Donchian Channels, Price Envelopes, VIX Fix, and Historical Volatility for
analyzing market volatility patterns and potential turning points.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union

from analysis_engine.analysis.advanced_ta.base import AdvancedAnalysisBase


class DonchianChannels(AdvancedAnalysisBase):
    """
    Donchian Channels analyzer
    
    Creates a channel based on the highest high and lowest low over a specified period,
    including an optional middle line as the average of the upper and lower bands.
    Useful for trend identification and breakout systems.
    """
    
    def __init__(
        self,
        name: str = "DonchianChannels",
        window: int = 20,
        include_middle: bool = True,
        high_col: str = "high",
        low_col: str = "low",
        **kwargs
    ):
        """Initialize the Donchian Channels analyzer.
        
        Args:
            name: Identifier for this analyzer
            window: Lookback period for channel calculation
            include_middle: Whether to calculate the middle line
            high_col: Column name for high prices
            low_col: Column name for low prices
            **kwargs: Additional parameters
        """
        super().__init__(name=name, **kwargs)
        self.window = window
        self.include_middle = include_middle
        self.high_col = high_col
        self.low_col = low_col
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Donchian Channels.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with Donchian Channels results
        """
        if self.high_col not in data.columns or self.low_col not in data.columns:
            raise ValueError(f"Data must contain '{self.high_col}' and '{self.low_col}' columns")
        
        # Calculate upper and lower bands
        upper = data[self.high_col].rolling(window=self.window).max()
        lower = data[self.low_col].rolling(window=self.window).min()
        
        results = {
            "dc_upper": upper,
            "dc_lower": lower
        }
        
        # Add middle line if requested
        if self.include_middle:
            middle = (upper + lower) / 2
            results["dc_middle"] = middle
        
        # Add breakout signals
        results["dc_breakout_up"] = (data["close"] > upper.shift(1)).astype(int)
        results["dc_breakout_down"] = (data["close"] < lower.shift(1)).astype(int)
        
        self.results = results
        return results


class PriceEnvelopes(AdvancedAnalysisBase):
    """
    Price Envelopes analyzer
    
    Creates bands around a moving average at a fixed percentage distance,
    useful for identifying overbought and oversold conditions.
    """
    
    def __init__(
        self,
        name: str = "PriceEnvelopes",
        window: int = 20,
        percent: float = 2.5,
        ma_type: str = "sma",
        column: str = "close",
        alert_threshold: Optional[float] = None,
        **kwargs
    ):
        """Initialize the Price Envelopes analyzer.
        
        Args:
            name: Identifier for this analyzer
            window: Lookback period for the moving average
            percent: Percentage for envelope width
            ma_type: Type of moving average ('sma', 'ema', 'wma')
            column: Column name for price data
            alert_threshold: Optional threshold for alerts
            **kwargs: Additional parameters
        """
        super().__init__(name=name, **kwargs)
        self.window = window
        self.percent = percent
        self.ma_type = ma_type.lower()
        self.column = column
        self.alert_threshold = alert_threshold
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Price Envelopes.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with Price Envelopes results
        """
        if self.column not in data.columns:
            raise ValueError(f"Data must contain '{self.column}' column")
        
        # Calculate the moving average
        if self.ma_type == "sma":
            ma = data[self.column].rolling(window=self.window).mean()
        elif self.ma_type == "ema":
            ma = data[self.column].ewm(span=self.window, adjust=False).mean()
        elif self.ma_type == "wma":
            weights = np.arange(1, self.window + 1)
            ma = data[self.column].rolling(window=self.window).apply(
                lambda x: np.sum(weights * x) / weights.sum(), raw=True
            )
        else:
            raise ValueError(f"Unsupported moving average type: {self.ma_type}")
        
        # Calculate upper and lower bands
        percent_factor = self.percent / 100.0
        upper = ma * (1 + percent_factor)
        lower = ma * (1 - percent_factor)
        
        results = {
            "pe_ma": ma,
            "pe_upper": upper,
            "pe_lower": lower
        }
        
        # Add alert signals if threshold is specified
        if self.alert_threshold is not None:
            threshold = self.alert_threshold / 100.0
            upper_threshold = ma * (1 + threshold)
            lower_threshold = ma * (1 - threshold)
            
            alert = np.zeros(len(data))
            alert[(data[self.column] > upper_threshold)] = 1
            alert[(data[self.column] < lower_threshold)] = -1
            
            results["pe_alert"] = pd.Series(alert, index=data.index)
            
        self.results = results
        return results


class VIXFix(AdvancedAnalysisBase):
    """
    VIX Fix Indicator analyzer
    
    A volatility indicator that measures relative volatility using a methodology
    similar to the VIX calculation. Useful for identifying potential market turns
    and volatility contraction/expansion cycles.
    """
    
    def __init__(
        self,
        name: str = "VIXFix",
        window: int = 22,
        smooth_window: int = 3,
        use_log: bool = True,
        filter_threshold: Optional[float] = None,
        **kwargs
    ):
        """Initialize the VIX Fix analyzer.
        
        Args:
            name: Identifier for this analyzer
            window: Lookback period for the volatility calculation
            smooth_window: Window for smoothing the values
            use_log: Whether to use logarithmic returns
            filter_threshold: Optional threshold for filtering signals
            **kwargs: Additional parameters
        """
        super().__init__(name=name, **kwargs)
        self.window = window
        self.smooth_window = smooth_window
        self.use_log = use_log
        self.filter_threshold = filter_threshold
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate VIX Fix Indicator.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with VIX Fix results
        """
        if 'close' not in data.columns or 'high' not in data.columns or 'low' not in data.columns:
            raise ValueError("Data must contain 'close', 'high', and 'low' columns")
        
        # Calculate true range
        tr = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                np.abs(data['high'] - data['close'].shift(1)),
                np.abs(data['low'] - data['close'].shift(1))
            )
        )
        
        # Calculate returns
        if self.use_log:
            returns = np.log(data['close'] / data['close'].shift(1))
        else:
            returns = data['close'].pct_change()
        
        # Calculate volatility
        vol = returns.rolling(window=self.window).std() * np.sqrt(252)  # Annualized
        
        # Calculate VIX Fix
        vix_fix = tr / data['close'] * 100 * np.sqrt(252)
        
        # Apply smoothing
        vix_fix_smooth = vix_fix.rolling(window=self.smooth_window).mean()
        
        results = {
            "vix_fix": vix_fix,
            "vix_fix_smooth": vix_fix_smooth
        }
        
        # Add signal based on threshold crossing if provided
        if self.filter_threshold is not None:
            signal = np.zeros(len(data))
            signal[vix_fix_smooth > self.filter_threshold] = 1
            signal[vix_fix_smooth < self.filter_threshold] = -1
            
            results["vix_fix_signal"] = pd.Series(signal, index=data.index)
        
        self.results = results
        return results


class HistoricalVolatility(AdvancedAnalysisBase):
    """
    Historical Volatility analyzer
    
    Calculates the standard deviation of price changes over a specified period,
    typically annualized. Useful for comparing volatility across markets or timeframes.
    """
    
    def __init__(
        self,
        name: str = "HistoricalVolatility",
        window: int = 21,
        periods: List[int] = None,
        annualize: bool = True,
        column: str = "close",
        scaling_factor: int = 252,  # Trading days in a year
        **kwargs
    ):
        """Initialize the Historical Volatility analyzer.
        
        Args:
            name: Identifier for this analyzer
            window: Primary lookback period for volatility calculation
            periods: Additional periods for multi-period comparison
            annualize: Whether to annualize the volatility
            column: Column name for price data
            scaling_factor: Number of periods in a year for annualization
            **kwargs: Additional parameters
        """
        super().__init__(name=name, **kwargs)
        self.window = window
        self.periods = periods if periods is not None else [window]
        if window not in self.periods:
            self.periods.append(window)
        self.annualize = annualize
        self.column = column
        self.scaling_factor = scaling_factor
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Historical Volatility.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with Historical Volatility results
        """
        if self.column not in data.columns:
            raise ValueError(f"Data must contain '{self.column}' column")
        
        # Calculate percentage changes
        pct_changes = data[self.column].pct_change()
        
        results = {}
        
        # Calculate volatility for each period
        for period in self.periods:
            # Standard deviation of percentage changes
            vol = pct_changes.rolling(window=period).std()
            
            # Annualize if requested
            if self.annualize:
                vol = vol * np.sqrt(self.scaling_factor)
            
            results[f"hist_vol_{period}"] = vol
            
        # Add volatility regime classification
        if len(data) > self.window * 3:
            # Calculate long-term average volatility and standard deviation
            long_term_vol = results[f"hist_vol_{self.window}"].rolling(window=self.window * 3).mean()
            long_term_vol_std = results[f"hist_vol_{self.window}"].rolling(window=self.window * 3).std()
            
            # Classify volatility regimes
            vol_z_score = (results[f"hist_vol_{self.window}"] - long_term_vol) / long_term_vol_std
            
            regime = np.zeros(len(data))
            # High volatility regime
            regime[vol_z_score > 1.0] = 1
            # Low volatility regime
            regime[vol_z_score < -1.0] = -1
            
            results["vol_regime"] = pd.Series(regime, index=data.index)
            
        self.results = results
        return results
