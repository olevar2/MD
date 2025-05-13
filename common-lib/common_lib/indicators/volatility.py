"""
Volatility Indicators Module.

This module provides implementations of various volatility-based indicators.
It is designed to be used across multiple services to ensure consistent indicator implementation.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from common_lib.indicators.base_indicator import BaseIndicator


class BollingerBands(BaseIndicator):
    """
    Bollinger Bands indicator.
    
    This volatility indicator creates bands around a moving average,
    with the width of the bands varying with volatility.
    """
    
    category = "volatility"
    name = "BollingerBands"
    default_params = {"window": 20, "num_std": 2.0, "column": "close"}
    required_params = {"window": int, "num_std": float, "column": str}
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Bollinger Bands indicator.
        
        Args:
            params: Dictionary of parameters for the indicator calculation.
                   If None, default_params will be used.
        """
        super().__init__(params)
        self.output_column_middle = f"bb_middle_{self.params['window']}"
        self.output_column_upper = f"bb_upper_{self.params['window']}_{self.params['num_std']}"
        self.output_column_lower = f"bb_lower_{self.params['window']}_{self.params['num_std']}"
        self.output_column_width = f"bb_width_{self.params['window']}_{self.params['num_std']}"
        self.output_column_pct_b = f"bb_pct_b_{self.params['window']}_{self.params['num_std']}"
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Bollinger Bands values
        """
        self.validate_input(data, [self.params["column"]])
        
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate middle band (SMA)
        result[self.output_column_middle] = result[self.params["column"]].rolling(window=self.params["window"]).mean()
        
        # Calculate standard deviation
        rolling_std = result[self.params["column"]].rolling(window=self.params["window"]).std()
        
        # Calculate upper and lower bands
        result[self.output_column_upper] = result[self.output_column_middle] + (rolling_std * self.params["num_std"])
        result[self.output_column_lower] = result[self.output_column_middle] - (rolling_std * self.params["num_std"])
        
        # Calculate bandwidth
        result[self.output_column_width] = (result[self.output_column_upper] - result[self.output_column_lower]) / result[self.output_column_middle]
        
        # Calculate %B
        result[self.output_column_pct_b] = (result[self.params["column"]] - result[self.output_column_lower]) / (result[self.output_column_upper] - result[self.output_column_lower])
        
        return result


class KeltnerChannels(BaseIndicator):
    """
    Keltner Channels indicator.
    
    This volatility indicator uses ATR to set channel width, creating
    a dynamic envelope around a moving average.
    """
    
    category = "volatility"
    name = "KeltnerChannels"
    default_params = {"window": 20, "atr_window": 10, "atr_multiplier": 2.0, "ma_method": "ema", "column": "close"}
    required_params = {"window": int, "atr_window": int, "atr_multiplier": float, "ma_method": str, "column": str}
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Keltner Channels indicator.
        
        Args:
            params: Dictionary of parameters for the indicator calculation.
                   If None, default_params will be used.
        """
        super().__init__(params)
        self.output_column_middle = f"kc_middle_{self.params['window']}_{self.params['ma_method']}"
        self.output_column_upper = f"kc_upper_{self.params['window']}_{self.params['atr_window']}_{self.params['atr_multiplier']}"
        self.output_column_lower = f"kc_lower_{self.params['window']}_{self.params['atr_window']}_{self.params['atr_multiplier']}"
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Keltner Channels for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Keltner Channels values
        """
        required_cols = ['high', 'low', 'close']
        self.validate_input(data, required_cols)
        
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate middle line
        if self.params["ma_method"] == 'sma':
            result[self.output_column_middle] = result[self.params["column"]].rolling(window=self.params["window"]).mean()
        elif self.params["ma_method"] == 'ema':
            result[self.output_column_middle] = result[self.params["column"]].ewm(span=self.params["window"], adjust=False).mean()
        else:
            raise ValueError(f"Invalid ma_method: {self.params['ma_method']}. Expected 'sma' or 'ema'.")
        
        # Calculate ATR
        high_low = result['high'] - result['low']
        high_close_prev = abs(result['high'] - result['close'].shift(1))
        low_close_prev = abs(result['low'] - result['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.params["atr_window"]).mean()
        
        # Calculate upper and lower bands
        result[self.output_column_upper] = result[self.output_column_middle] + (atr * self.params["atr_multiplier"])
        result[self.output_column_lower] = result[self.output_column_middle] - (atr * self.params["atr_multiplier"])
        
        return result


class DonchianChannels(BaseIndicator):
    """
    Donchian Channels indicator.
    
    This volatility indicator shows the highest high and lowest low over a specified period,
    creating channels that indicate price extremes and potential breakouts.
    """
    
    category = "volatility"
    name = "DonchianChannels"
    default_params = {"window": 20}
    required_params = {"window": int}
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Donchian Channels indicator.
        
        Args:
            params: Dictionary of parameters for the indicator calculation.
                   If None, default_params will be used.
        """
        super().__init__(params)
        self.output_column_upper = f"donchian_upper_{self.params['window']}"
        self.output_column_lower = f"donchian_lower_{self.params['window']}"
        self.output_column_middle = f"donchian_middle_{self.params['window']}"
        self.output_column_width = f"donchian_width_{self.params['window']}"
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Donchian Channels for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Donchian Channels values
        """
        required_cols = ['high', 'low']
        self.validate_input(data, required_cols)
        
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate channel components
        result[self.output_column_upper] = result['high'].rolling(window=self.params["window"]).max()
        result[self.output_column_lower] = result['low'].rolling(window=self.params["window"]).min()
        result[self.output_column_middle] = (result[self.output_column_upper] + result[self.output_column_lower]) / 2
        result[self.output_column_width] = result[self.output_column_upper] - result[self.output_column_lower]
        
        return result


class AverageTrueRange(BaseIndicator):
    """
    Average True Range (ATR) indicator.
    
    This volatility indicator measures market volatility by decomposing the
    entire range of an asset price for a period.
    """
    
    category = "volatility"
    name = "AverageTrueRange"
    default_params = {"window": 14}
    required_params = {"window": int}
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Average True Range indicator.
        
        Args:
            params: Dictionary of parameters for the indicator calculation.
                   If None, default_params will be used.
        """
        super().__init__(params)
        self.output_column = f"atr_{self.params['window']}"
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ATR for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with ATR values
        """
        required_cols = ['high', 'low', 'close']
        self.validate_input(data, required_cols)
        
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate true range
        high_low = result['high'] - result['low']
        high_close_prev = abs(result['high'] - result['close'].shift(1))
        low_close_prev = abs(result['low'] - result['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate ATR using simple moving average
        result[self.output_column] = true_range.rolling(window=self.params["window"]).mean()
        
        return result


class PriceEnvelopes(BaseIndicator):
    """
    Price Envelopes indicator.
    
    This volatility indicator creates percentage-based bands around a moving average,
    helping to identify extreme overbought or oversold conditions.
    """
    
    category = "volatility"
    name = "PriceEnvelopes"
    default_params = {"window": 20, "percent": 2.5, "ma_method": "sma", "column": "close"}
    required_params = {"window": int, "percent": float, "ma_method": str, "column": str}
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Price Envelopes indicator.
        
        Args:
            params: Dictionary of parameters for the indicator calculation.
                   If None, default_params will be used.
        """
        super().__init__(params)
        self.output_column_middle = f"env_middle_{self.params['window']}_{self.params['ma_method']}"
        self.output_column_upper = f"env_upper_{self.params['window']}_{self.params['percent']}"
        self.output_column_lower = f"env_lower_{self.params['window']}_{self.params['percent']}"
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Price Envelopes for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Price Envelopes values
        """
        self.validate_input(data, [self.params["column"]])
        
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate middle line (moving average)
        if self.params["ma_method"] == 'sma':
            result[self.output_column_middle] = result[self.params["column"]].rolling(window=self.params["window"]).mean()
        elif self.params["ma_method"] == 'ema':
            result[self.output_column_middle] = result[self.params["column"]].ewm(span=self.params["window"], adjust=False).mean()
        else:
            raise ValueError(f"Invalid ma_method: {self.params['ma_method']}. Expected 'sma' or 'ema'.")
        
        # Calculate upper and lower bands
        percent_factor = self.params["percent"] / 100
        result[self.output_column_upper] = result[self.output_column_middle] * (1 + percent_factor)
        result[self.output_column_lower] = result[self.output_column_middle] * (1 - percent_factor)
        
        return result


class HistoricalVolatility(BaseIndicator):
    """
    Historical Volatility indicator.
    
    This indicator calculates the standard deviation of price changes over a specified period,
    providing a measure of market volatility.
    """
    
    category = "volatility"
    name = "HistoricalVolatility"
    default_params = {"window": 20, "column": "close", "annualize": True, "trading_periods": 252}
    required_params = {"window": int, "column": str, "annualize": bool}
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Historical Volatility indicator.
        
        Args:
            params: Dictionary of parameters for the indicator calculation.
                   If None, default_params will be used.
        """
        super().__init__(params)
        self.output_column = f"hv_{self.params['window']}"
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Historical Volatility for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Historical Volatility values
        """
        self.validate_input(data, [self.params["column"]])
        
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate daily returns
        returns = result[self.params["column"]].pct_change()
        
        # Calculate rolling standard deviation of returns
        volatility = returns.rolling(window=self.params["window"]).std()
        
        # Annualize if requested
        if self.params["annualize"]:
            trading_periods = self.params.get("trading_periods", 252)  # Default to 252 trading days per year
            volatility = volatility * np.sqrt(trading_periods)
        
        result[self.output_column] = volatility
        
        return result