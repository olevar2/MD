"""
Timeframe Confluence Indicator implementation.

This module implements a system to measure indicator concordance 
across multiple timeframes with visualization capabilities.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
from enum import Enum

from feature_store_service.indicators.base_indicator import BaseIndicator


class SignalType(Enum):
    """Enum for signal types."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class TimeframeConfluenceIndicator(BaseIndicator):
    """
    Timeframe Confluence Indicator.
    
    This indicator measures the concordance of signals across multiple timeframes
    for any technical indicator or combination of indicators.
    """
    
    category = "multi_timeframe_concordance"
    
    def __init__(
        self,
        signal_functions: Dict[str, Callable],
        timeframes: List[str],
        reference_timeframe: Optional[str] = None,
        concordance_window: int = 1,
        **kwargs
    ):
        """
        Initialize Timeframe Confluence Indicator.
        
        Args:
            signal_functions: Dictionary of signal functions, each returning 1 (bullish), 
                             -1 (bearish), or 0 (neutral) for a given dataframe
            timeframes: List of timeframes to analyze
            reference_timeframe: Timeframe to align results to (defaults to lowest timeframe)
            concordance_window: Number of bars to check for confluence
            **kwargs: Additional parameters
        """
        self.signal_functions = signal_functions
        self.timeframes = timeframes
        self.reference_timeframe = reference_timeframe if reference_timeframe else timeframes[0]
        self.concordance_window = concordance_window
        
        self.name = "timeframe_confluence"
        
    def calculate(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate Timeframe Confluence for the given data.
        
        Args:
            data: Dictionary of DataFrames with OHLCV data for each timeframe
            
        Returns:
            DataFrame with Timeframe Confluence indicator values
        """
        # Check if all required timeframes are available
        for tf in self.timeframes:
            if tf not in data:
                raise ValueError(f"Data must contain '{tf}' timeframe data")
        
        # Use reference timeframe as the base result
        result = data[self.reference_timeframe].copy()
        
        # Calculate signals for each indicator and timeframe
        signals = {}
        for indicator_name, signal_func in self.signal_functions.items():
            signals[indicator_name] = {}
            
            for tf in self.timeframes:
                tf_data = data[tf].copy()
                
                # Calculate the signal (-1, 0, or 1)
                signal_col = f"{indicator_name}_{tf}_signal"
                result[signal_col] = signal_func(tf_data)
                signals[indicator_name][tf] = signal_col
        
        # Calculate concordance for each indicator
        for indicator_name, tf_signals in signals.items():
            self._calculate_indicator_concordance(result, indicator_name, tf_signals)
            
        # Calculate overall concordance across all indicators
        self._calculate_overall_concordance(result, signals)
        
        return result
    
    def _calculate_indicator_concordance(
        self, 
        result: pd.DataFrame, 
        indicator_name: str, 
        tf_signals: Dict[str, str]
    ) -> None:
        """
        Calculate concordance for a specific indicator across timeframes.
        
        Args:
            result: DataFrame to store concordance
            indicator_name: Name of the indicator
            tf_signals: Dictionary mapping timeframes to signal column names
        """
        # Count bullish, bearish, and neutral signals
        bullish_signals = pd.DataFrame(index=result.index)
        bearish_signals = pd.DataFrame(index=result.index)
        neutral_signals = pd.DataFrame(index=result.index)
        
        for tf, col in tf_signals.items():
            bullish_signals[tf] = (result[col] > 0).astype(int)
            bearish_signals[tf] = (result[col] < 0).astype(int)
            neutral_signals[tf] = (result[col] == 0).astype(int)
            
        # Calculate concordance percentage for each signal type
        result[f"{indicator_name}_bullish_concordance"] = bullish_signals.sum(axis=1) * 100 / len(tf_signals)
        result[f"{indicator_name}_bearish_concordance"] = bearish_signals.sum(axis=1) * 100 / len(tf_signals)
        result[f"{indicator_name}_neutral_concordance"] = neutral_signals.sum(axis=1) * 100 / len(tf_signals)
        
        # Determine the dominant signal type
        result[f"{indicator_name}_signal_type"] = 0
        result.loc[result[f"{indicator_name}_bullish_concordance"] > 
                 result[f"{indicator_name}_bearish_concordance"], f"{indicator_name}_signal_type"] = 1
        result.loc[result[f"{indicator_name}_bearish_concordance"] > 
                 result[f"{indicator_name}_bullish_concordance"], f"{indicator_name}_signal_type"] = -1
                 
        # Calculate overall concordance for this indicator (0-100%)
        result[f"{indicator_name}_concordance"] = result[[
            f"{indicator_name}_bullish_concordance", 
            f"{indicator_name}_bearish_concordance", 
            f"{indicator_name}_neutral_concordance"
        ]].max(axis=1)
    
    def _calculate_overall_concordance(
        self, 
        result: pd.DataFrame, 
        signals: Dict[str, Dict[str, str]]
    ) -> None:
        """
        Calculate overall concordance across all indicators and timeframes.
        
        Args:
            result: DataFrame to store concordance
            signals: Nested dictionary of indicator names to timeframes to signal columns
        """
        # Get all individual indicator concordance columns
        concordance_cols = [col for col in result.columns if col.endswith("_concordance") 
                           and not col.endswith(("_bullish_concordance", "_bearish_concordance", "_neutral_concordance"))]
        
        # Calculate the mean concordance across all indicators
        result["overall_concordance"] = result[concordance_cols].mean(axis=1)
        
        # Get all signal type columns
        signal_type_cols = [col for col in result.columns if col.endswith("_signal_type")]
        
        # Count number of bullish and bearish signals
        bullish_count = (result[signal_type_cols] > 0).sum(axis=1)
        bearish_count = (result[signal_type_cols] < 0).sum(axis=1)
        
        # Calculate overall signal concordance
        total_signals = len(signal_type_cols)
        result["overall_bullish_agreement"] = bullish_count * 100 / total_signals
        result["overall_bearish_agreement"] = bearish_count * 100 / total_signals
        
        # Overall signal direction
        result["overall_signal"] = 0
        result.loc[bullish_count > bearish_count, "overall_signal"] = 1
        result.loc[bearish_count > bullish_count, "overall_signal"] = -1
        
    def get_visual_analysis_data(self, result: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate data for advanced visual analysis of concordance.
        
        Args:
            result: DataFrame with calculated concordance
            
        Returns:
            Dictionary with visualization data
        """
        indicators = list(self.signal_functions.keys())
        
        visualization_data = {
            "timeframes": self.timeframes,
            "indicators": indicators,
            "overall_concordance": result["overall_concordance"].tolist(),
            "overall_signal": result["overall_signal"].tolist(),
            "indicator_data": {}
        }
        
        for indicator in indicators:
            visualization_data["indicator_data"][indicator] = {
                "concordance": result[f"{indicator}_concordance"].tolist(),
                "signal_type": result[f"{indicator}_signal_type"].tolist(),
                "bullish_concordance": result[f"{indicator}_bullish_concordance"].tolist(),
                "bearish_concordance": result[f"{indicator}_bearish_concordance"].tolist()
            }
            
        return visualization_data


# Example signal functions
def rsi_signal(data: pd.DataFrame, column: str = "rsi_14", overbought: float = 70, oversold: float = 30) -> pd.Series:
    """Generate RSI signals: 1 for bullish (oversold), -1 for bearish (overbought), 0 for neutral."""
    signal = pd.Series(0, index=data.index)
    signal[data[column] < oversold] = 1
    signal[data[column] > overbought] = -1
    return signal

def macd_signal(data: pd.DataFrame, line_col: str = "macd_line", signal_col: str = "macd_signal") -> pd.Series:
    """Generate MACD signals: 1 for bullish (line crosses above signal), -1 for bearish (line crosses below signal)."""
    signal = pd.Series(0, index=data.index)
    signal[data[line_col] > data[signal_col]] = 1
    signal[data[line_col] < data[signal_col]] = -1
    return signal

def ma_cross_signal(data: pd.DataFrame, fast_ma: str = "sma_20", slow_ma: str = "sma_50") -> pd.Series:
    """Generate MA crossover signals: 1 for bullish (fast MA above slow MA), -1 for bearish (fast MA below slow MA)."""
    signal = pd.Series(0, index=data.index)
    signal[data[fast_ma] > data[slow_ma]] = 1
    signal[data[fast_ma] < data[slow_ma]] = -1
    return signal
"""
