"""
Trend Indicators Module

This module provides implementations for various trend-related technical indicators:
- Parabolic SAR
- And others

These indicators help identify the direction of market trends and potential
reversal points.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, List

from analysis_engine.analysis.basic_ta.base import BaseIndicator
from analysis_engine.utils.validation import validate_dataframe

logger = logging.getLogger(__name__)


class ParabolicSAR(BaseIndicator):
    """
    Parabolic Stop and Reverse (SAR) indicator.
    
    The Parabolic SAR is used to determine trend direction and potential reversal points.
    It places dots on a chart that indicate potential reversals in price movement.
    
    When the price is above the dots, it's generally a bullish signal.
    When price is below the dots, it's generally a bearish signal.
    
    Implementation follows Wilder's original methodology.
    """
    
    def __init__(
        self,
        initial_af: float = 0.02,
        max_af: float = 0.2,
        af_step: float = 0.02,
        high_column: str = "high",
        low_column: str = "low",
        **kwargs
    ):
        """
        Initialize Parabolic SAR indicator.
        
        Args:
            initial_af: Initial acceleration factor
            max_af: Maximum acceleration factor
            af_step: Acceleration factor step
            high_column: Column name for high prices
            low_column: Column name for low prices
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.initial_af = initial_af
        self.max_af = max_af
        self.af_step = af_step
        self.high_column = high_column
        self.low_column = low_column
        
        # Output column name
        self.output_column = "psar"
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Parabolic SAR for the given data.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with Parabolic SAR values
        """
        validate_dataframe(df, required_columns=[self.high_column, self.low_column])
        
        result_df = df.copy()
        
        # Get high and low prices
        high = result_df[self.high_column].values
        low = result_df[self.low_column].values
        
        # Initialize arrays
        psar = np.zeros(len(high))
        psar[:] = np.nan
        
        # Start with bullish trend by default
        bull = True
        
        # Initialize values for first iteration
        af = self.initial_af
        extreme_point = high[0]
        psar[0] = low[0]  # Start below first candle for bullish trend
        
        # Calculate Parabolic SAR for each bar
        for i in range(1, len(high)):
            # Previous SAR value
            psar_prev = psar[i-1]
            
            # Calculate new SAR value
            if bull:
                # Bullish trend
                psar[i] = psar_prev + af * (extreme_point - psar_prev)
                
                # Ensure SAR is below the lows of the previous two periods
                psar[i] = min(psar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
                
                # Check for trend reversal
                if psar[i] > low[i]:
                    bull = False
                    psar[i] = extreme_point
                    extreme_point = low[i]
                    af = self.initial_af
                else:
                    # Continue bullish trend
                    if high[i] > extreme_point:
                        extreme_point = high[i]
                        af = min(af + self.af_step, self.max_af)
            else:
                # Bearish trend
                psar[i] = psar_prev + af * (extreme_point - psar_prev)
                
                # Ensure SAR is above the highs of the previous two periods
                psar[i] = max(psar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
                
                # Check for trend reversal
                if psar[i] < high[i]:
                    bull = True
                    psar[i] = extreme_point
                    extreme_point = high[i]
                    af = self.initial_af
                else:
                    # Continue bearish trend
                    if low[i] < extreme_point:
                        extreme_point = low[i]
                        af = min(af + self.af_step, self.max_af)
        
        # Add Parabolic SAR to result DataFrame
        result_df[self.output_column] = psar
        
        # Add trend column (1 for bullish, -1 for bearish)
        result_df["psar_trend"] = np.where(high > psar, 1, -1)
        
        return result_df
        
    def initialize_incremental(self) -> Dict[str, Any]:
        """Initialize state for incremental calculation."""
        return {
            "bull": True,
            "af": self.initial_af,
            "extreme_point": 0,
            "previous_psar": 0,
            "previous_high": 0,
            "previous_low": 0,
            "first_run": True
        }
        
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update state with new data point."""
        current_high = new_data.get(self.high_column, 0)
        current_low = new_data.get(self.low_column, 0)
        
        # Initialize values on first run
        if state["first_run"]:
            state["extreme_point"] = current_high
            state["previous_psar"] = current_low
            state["previous_high"] = current_high
            state["previous_low"] = current_low
            state["first_run"] = False
            return state
        
        bull = state["bull"]
        af = state["af"]
        extreme_point = state["extreme_point"]
        psar_prev = state["previous_psar"]
        
        # Calculate new SAR value
        if bull:
            # Bullish trend
            psar = psar_prev + af * (extreme_point - psar_prev)
            
            # Ensure SAR is below the previous low
            psar = min(psar, state["previous_low"])
            
            # Check for trend reversal
            if psar > current_low:
                bull = False
                psar = extreme_point
                extreme_point = current_low
                af = self.initial_af
            else:
                # Continue bullish trend
                if current_high > extreme_point:
                    extreme_point = current_high
                    af = min(af + self.af_step, self.max_af)
        else:
            # Bearish trend
            psar = psar_prev + af * (extreme_point - psar_prev)
            
            # Ensure SAR is above the previous high
            psar = max(psar, state["previous_high"])
            
            # Check for trend reversal
            if psar < current_high:
                bull = True
                psar = extreme_point
                extreme_point = current_high
                af = self.initial_af
            else:
                # Continue bearish trend
                if current_low < extreme_point:
                    extreme_point = current_low
                    af = min(af + self.af_step, self.max_af)
        
        # Update state
        state["bull"] = bull
        state["af"] = af
        state["extreme_point"] = extreme_point
        state["previous_psar"] = psar
        state["previous_high"] = current_high
        state["previous_low"] = current_low
        state["last_psar"] = psar
        state["last_psar_trend"] = 1 if bull else -1
        
        return state
