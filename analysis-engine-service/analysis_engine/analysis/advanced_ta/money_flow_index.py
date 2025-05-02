"""
Money Flow Index (MFI) Indicator Module

This module implements the Money Flow Index (MFI) indicator, a momentum oscillator that
uses price and volume to measure buying and selling pressure. MFI is also known as 
volume-weighted RSI as it incorporates volume, unlike the standard RSI.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, List

from analysis_engine.analysis.advanced_ta.base import AdvancedAnalysisBase
from analysis_engine.utils.validation import validate_dataframe

logger = logging.getLogger(__name__)


class MoneyFlowIndex(AdvancedAnalysisBase):
    """
    Money Flow Index (MFI) Indicator
    
    MFI is a momentum oscillator that measures the inflow and outflow of money into an asset
    over a specific period of time. It's used to identify overbought or oversold conditions
    and potential price reversals.
    
    The indicator helps identify:
    - Overbought conditions (typically above 80)
    - Oversold conditions (typically below 20)
    - Divergences between price and money flow
    - Potential trend reversals
    """
    
    def __init__(
        self,
        period: int = 14,
        overbought_level: float = 80.0,
        oversold_level: float = 20.0,
        output_prefix: str = "MFI",
        **kwargs
    ):
        """
        Initialize Money Flow Index indicator.
        
        Args:
            period: The lookback period for calculating MFI
            overbought_level: Level above which the asset is considered overbought
            oversold_level: Level below which the asset is considered oversold
            output_prefix: Prefix for output column names
            **kwargs: Additional parameters
        """
        parameters = {
            "period": period,
            "overbought_level": overbought_level,
            "oversold_level": oversold_level,
            "output_prefix": output_prefix
        }
        super().__init__("Money Flow Index", parameters)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Money Flow Index (MFI) for the given data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with MFI values
        """
        # Input validation
        validate_dataframe(df, required_columns=["high", "low", "close", "volume"])
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Extract parameters
        period = self.parameters["period"]
        overbought = self.parameters["overbought_level"]
        oversold = self.parameters["oversold_level"]
        prefix = self.parameters["output_prefix"]
        
        # Calculate typical price
        result_df["typical_price"] = (result_df["high"] + result_df["low"] + result_df["close"]) / 3.0
        
        # Calculate raw money flow
        result_df["money_flow"] = result_df["typical_price"] * result_df["volume"]
        
        # Calculate price direction based on typical price change
        result_df["direction"] = result_df["typical_price"].diff().fillna(0.0)
        
        # Calculate positive and negative money flow
        result_df["positive_flow"] = np.where(result_df["direction"] >= 0, result_df["money_flow"], 0.0)
        result_df["negative_flow"] = np.where(result_df["direction"] < 0, result_df["money_flow"], 0.0)
        
        # Calculate rolling sums of positive and negative flows
        result_df["pos_flow_sum"] = result_df["positive_flow"].rolling(window=period).sum()
        result_df["neg_flow_sum"] = result_df["negative_flow"].rolling(window=period).sum()
        
        # Calculate money flow ratio and MFI
        # Avoid division by zero
        result_df["money_ratio"] = np.where(
            result_df["neg_flow_sum"] != 0,
            result_df["pos_flow_sum"] / result_df["neg_flow_sum"],
            100.0
        )
        
        # MFI calculation
        result_df[f"{prefix}"] = 100.0 - (100.0 / (1.0 + result_df["money_ratio"]))
        
        # Add overbought/oversold signals
        result_df[f"{prefix}_overbought"] = result_df[f"{prefix}"] >= overbought
        result_df[f"{prefix}_oversold"] = result_df[f"{prefix}"] <= oversold
        
        # Drop temporary columns
        cols_to_drop = ["typical_price", "money_flow", "direction", "positive_flow", 
                        "negative_flow", "pos_flow_sum", "neg_flow_sum", "money_ratio"]
        result_df = result_df.drop(columns=cols_to_drop)
        
        return result_df
    
    def initialize_incremental(self) -> Dict[str, Any]:
        """
        Initialize state for incremental calculation
        
        Returns:
            Initial state dictionary
        """
        state = {
            "period": self.parameters["period"],
            "last_typical_price": None,
            "positive_flows": [],
            "negative_flows": [],
            "overbought_level": self.parameters["overbought_level"],
            "oversold_level": self.parameters["oversold_level"]
        }
        return state
    
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update MFI calculation with new data incrementally
        
        Args:
            state: Current calculation state
            new_data: New data point
            
        Returns:
            Updated state and MFI values
        """
        # Validate new data
        required_fields = ["high", "low", "close", "volume"]
        if not all(field in new_data for field in required_fields):
            logger.warning(f"Required fields {required_fields} missing in new_data")
            return state
        
        # Calculate typical price
        typical_price = (new_data["high"] + new_data["low"] + new_data["close"]) / 3.0
        
        # Calculate raw money flow
        money_flow = typical_price * new_data["volume"]
        
        # Determine direction
        direction = 0.0
        if state["last_typical_price"] is not None:
            direction = typical_price - state["last_typical_price"]
        
        # Calculate positive or negative flow
        if direction >= 0:
            positive_flow = money_flow
            negative_flow = 0.0
        else:
            positive_flow = 0.0
            negative_flow = money_flow
        
        # Update flow buffers
        state["positive_flows"].append(positive_flow)
        state["negative_flows"].append(negative_flow)
        
        # Keep buffers at the right size
        period = state["period"]
        if len(state["positive_flows"]) > period:
            state["positive_flows"] = state["positive_flows"][-period:]
        if len(state["negative_flows"]) > period:
            state["negative_flows"] = state["negative_flows"][-period:]
        
        # Calculate MFI
        pos_flow_sum = sum(state["positive_flows"])
        neg_flow_sum = sum(state["negative_flows"])
        
        if neg_flow_sum == 0:
            money_ratio = 100.0
        else:
            money_ratio = pos_flow_sum / neg_flow_sum
        
        mfi_value = 100.0 - (100.0 / (1.0 + money_ratio))
        
        # Update state
        state["last_typical_price"] = typical_price
        state["current_mfi"] = mfi_value
        state["is_overbought"] = mfi_value >= state["overbought_level"]
        state["is_oversold"] = mfi_value <= state["oversold_level"]
        
        return state
