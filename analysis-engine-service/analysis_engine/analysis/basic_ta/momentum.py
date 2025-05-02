"""
Momentum Indicators Module

This module provides implementations for various momentum-based technical indicators:
- Momentum
- Rate of Change (ROC)
- And others

These indicators help identify the strength or weakness of a price trend.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, List

from analysis_engine.analysis.basic_ta.base import BaseIndicator
from analysis_engine.utils.validation import validate_dataframe

logger = logging.getLogger(__name__)


class Momentum(BaseIndicator):
    """
    Momentum indicator.
    
    The Momentum indicator measures the amount that a security's price has changed
    over a given time period. It compares the current price with the price of n periods ago.
    
    Formula:
        Momentum = Current Price - Price n periods ago
    """
    
    def __init__(
        self,
        period: int = 14,
        price_column: str = "close",
        normalize: bool = False,
        **kwargs
    ):
        """
        Initialize Momentum indicator.
        
        Args:
            period: Lookback period
            price_column: Column name for price data
            normalize: Whether to normalize the result (divide by price n periods ago)
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.period = period
        self.price_column = price_column
        self.normalize = normalize
        
        # Output column name
        self.output_column = f"momentum_{period}"
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Momentum for the given data.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with Momentum values
        """
        validate_dataframe(df, required_columns=[self.price_column])
        
        result_df = df.copy()
        
        # Calculate price difference
        if self.normalize:
            # Normalize by price n periods ago (percentage change)
            momentum = result_df[self.price_column].pct_change(periods=self.period) * 100
        else:
            # Absolute price difference
            momentum = result_df[self.price_column].diff(periods=self.period)
        
        # Add Momentum to result DataFrame
        result_df[self.output_column] = momentum
        
        return result_df
        
    def initialize_incremental(self) -> Dict[str, Any]:
        """Initialize state for incremental calculation."""
        return {
            "prices": [],
            "last_momentum": None
        }
        
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update state with new data point."""
        price = new_data.get(self.price_column, 0)
        
        # Update price buffer
        state["prices"].append(price)
        
        # Keep only the needed historical data
        if len(state["prices"]) > self.period + 1:
            state["prices"] = state["prices"][-(self.period+1):]
        
        # Calculate Momentum if we have enough data
        if len(state["prices"]) > self.period:
            current_price = state["prices"][-1]
            past_price = state["prices"][-self.period-1]
            
            if self.normalize and past_price != 0:
                state["last_momentum"] = ((current_price / past_price) - 1) * 100
            else:
                state["last_momentum"] = current_price - past_price
        
        return state


class RateOfChange(BaseIndicator):
    """
    Rate of Change (ROC) indicator.
    
    The Rate of Change indicator measures the percentage change in price between
    the current price and the price n periods ago.
    
    Formula:
        ROC = ((Current Price - Price n periods ago) / Price n periods ago) * 100
    """
    
    def __init__(
        self,
        period: int = 14,
        price_column: str = "close",
        smoothing_period: Optional[int] = None,
        smoothing_type: str = "sma",
        **kwargs
    ):
        """
        Initialize Rate of Change indicator.
        
        Args:
            period: Lookback period
            price_column: Column name for price data
            smoothing_period: Optional period for smoothing the ROC
            smoothing_type: Type of moving average for smoothing ('sma', 'ema')
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.period = period
        self.price_column = price_column
        self.smoothing_period = smoothing_period
        self.smoothing_type = smoothing_type
        
        # Output column name
        self.output_column = f"roc_{period}"
        if smoothing_period:
            self.smoothed_column = f"{self.output_column}_{smoothing_type}{smoothing_period}"
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Rate of Change for the given data.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with ROC values
        """
        validate_dataframe(df, required_columns=[self.price_column])
        
        result_df = df.copy()
        
        # Calculate percentage change
        roc = result_df[self.price_column].pct_change(periods=self.period) * 100
        
        # Add ROC to result DataFrame
        result_df[self.output_column] = roc
        
        # Apply smoothing if requested
        if self.smoothing_period:
            if self.smoothing_type == 'ema':
                smooth_roc = roc.ewm(span=self.smoothing_period, adjust=False).mean()
            else:  # Default to SMA
                smooth_roc = roc.rolling(window=self.smoothing_period).mean()
                
            result_df[self.smoothed_column] = smooth_roc
        
        return result_df
        
    def initialize_incremental(self) -> Dict[str, Any]:
        """Initialize state for incremental calculation."""
        state = {
            "prices": [],
            "roc_values": [],
            "last_roc": None
        }
        
        if self.smoothing_period:
            state["last_smoothed_roc"] = None
            
        return state
        
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update state with new data point."""
        price = new_data.get(self.price_column, 0)
        
        # Update price buffer
        state["prices"].append(price)
        
        # Keep only the needed historical data
        max_periods = max(self.period, 0) + (self.smoothing_period or 0)
        if len(state["prices"]) > max_periods + 1:
            state["prices"] = state["prices"][-(max_periods+1):]
        
        # Calculate ROC if we have enough data
        if len(state["prices"]) > self.period:
            current_price = state["prices"][-1]
            past_price = state["prices"][-self.period-1]
            
            if past_price != 0:
                roc = ((current_price / past_price) - 1) * 100
                state["last_roc"] = roc
                
                # Update ROC values buffer for smoothing
                state["roc_values"].append(roc)
                
                # Keep only needed ROC values for smoothing
                if self.smoothing_period and len(state["roc_values"]) > self.smoothing_period:
                    state["roc_values"] = state["roc_values"][-self.smoothing_period:]
                
                # Calculate smoothed ROC if applicable
                if self.smoothing_period and len(state["roc_values"]) == self.smoothing_period:
                    if self.smoothing_type == 'ema':
                        # Simple implementation of EMA
                        alpha = 2 / (self.smoothing_period + 1)
                        if state["last_smoothed_roc"] is not None:
                            state["last_smoothed_roc"] = (alpha * roc) + ((1 - alpha) * state["last_smoothed_roc"])
                        else:
                            state["last_smoothed_roc"] = np.mean(state["roc_values"])
                    else:
                        # Simple moving average
                        state["last_smoothed_roc"] = np.mean(state["roc_values"])
            else:
                state["last_roc"] = 0
        
        return state
