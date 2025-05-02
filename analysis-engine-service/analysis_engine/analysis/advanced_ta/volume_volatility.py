"""
Volume and Volatility Analysis Module

This module provides comprehensive volume and volatility analysis tools including:
- Volume Profile
- Market Profile (Time/Price Opportunity)
- Average True Range (ATR) and variations
- Volume-Weighted Average Price (VWAP)
- Money Flow Index (MFI)
- On-Balance Volume (OBV)
- Chaikin Money Flow (CMF)
- Accumulation/Distribution Line
- Volume Delta
- Relative Volume
- Force Index
- Ease of Movement
- Elder Force Index
- Klinger Volume Oscillator
- Volatility Bands/Envelopes
- Volatility Ratio
- Liquidity Analysis

Implementations include both standard and incremental calculation approaches.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from datetime import datetime, timedelta
import math
from enum import Enum
from collections import defaultdict
import logging

from analysis_engine.analysis.advanced_ta.base import (
    AdvancedAnalysisBase,
    ConfidenceLevel,
    MarketDirection,
    AnalysisTimeframe
)
from analysis_engine.core.base.components import AdvancedAnalysisBase as CoreAdvancedAnalysisBase
from analysis_engine.analysis.confluence.confluence_analyzer import ConfluenceAnalyzer

logger = logging.getLogger(__name__)

class MarketRegimeAnalyzer(AdvancedAnalysisBase):
    """
    Market Regime Detector
    
    Uses volatility and volume metrics (assumed pre-calculated) to detect market regimes.
    """
    
    class MarketRegime(Enum):
        TRENDING_UP = "Trending Up"
        TRENDING_DOWN = "Trending Down"
        RANGING = "Ranging"
        VOLATILE = "Volatile"
        BREAKOUT_UP = "Breakout Up"
        BREAKOUT_DOWN = "Breakout Down"
        EXHAUSTION = "Exhaustion"
        EXPANSION = "Expansion"
        CONTRACTION = "Contraction"
        UNDEFINED = "Undefined"
    
    def __init__(self, lookback_period: int = 20, 
                 atr_period: int = 14, # Keep atr_period for referencing the correct ATR column
                 regime_change_threshold: float = 0.7):
        """
        Initialize the Market Regime Detector
        
        Args:
            lookback_period: Period for calculating market metrics
            atr_period: Period for referencing the pre-calculated ATR column (e.g., 'ATR_14')
            regime_change_threshold: Threshold for regime change confirmation
        """
        parameters = {
            "lookback_period": lookback_period,
            "atr_period": atr_period,
            "regime_change_threshold": regime_change_threshold
        }
        super().__init__("Market Regime Detector", parameters)
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regimes for the given data (assumes ATR is pre-calculated)
        
        Args:
            df: DataFrame with OHLCV data and pre-calculated ATR (e.g., 'ATR_14')
            
        Returns:
            DataFrame with market regime labels and metrics
        """
        # Ensure we have required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        atr_col = f"ATR_{self.parameters['atr_period']}" # Construct expected ATR column name
        required_cols.append(atr_col)
        
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"DataFrame must contain columns: {required_cols}. Missing: {missing}")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # ATR is assumed to be present in result_df[atr_col]
        
        # Calculate directional movement
        result_df["price_change"] = result_df["close"].diff()
        result_df["price_pct_change"] = result_df["close"].pct_change()
        
        # Calculate rolling metrics
        lookback = self.parameters["lookback_period"]
        result_df["rolling_mean"] = result_df["close"].rolling(window=lookback).mean()
        result_df["rolling_std"] = result_df["close"].rolling(window=lookback).std()
        result_df["rolling_volume_mean"] = result_df["volume"].rolling(window=lookback).mean()
        result_df["volume_ratio"] = result_df["volume"] / result_df["rolling_volume_mean"]
        result_df["price_distance"] = (result_df["close"] - result_df["rolling_mean"]) / result_df["rolling_std"]
        
        # Calculate directionality (ADX-like)
        result_df["up_move"] = np.maximum(result_df["high"] - result_df["high"].shift(1), 0)
        result_df["down_move"] = np.maximum(result_df["low"].shift(1) - result_df["low"], 0)
        
        # Filter for significant moves only
        atr_threshold = 0.2  # Use only moves that are at least 20% of ATR
        result_df["up_move"] = np.where(
            result_df["up_move"] > result_df[atr_col] * atr_threshold,
            result_df["up_move"],
            0
        )
        result_df["down_move"] = np.where(
            result_df["down_move"] > result_df[atr_col] * atr_threshold,
            result_df["down_move"],
            0
        )
        
        # Calculate smoothed directional indicators
        result_df["smoothed_up"] = result_df["up_move"].rolling(window=lookback).sum()
        result_df["smoothed_down"] = result_df["down_move"].rolling(window=lookback).sum()
        # Avoid division by zero
        denominator = (result_df["smoothed_up"] + result_df["smoothed_down"])
        result_df["direction_strength"] = np.where(
            denominator != 0, 
            abs(result_df["smoothed_up"] - result_df["smoothed_down"]) / denominator, 
            0
        )
        
        # Fill NAs with 0
        result_df["direction_strength"].fillna(0, inplace=True)
        
        # Volatility relative to history
        rolling_atr_mean = result_df[atr_col].rolling(window=lookback*2).mean()
        result_df["volatility_ratio"] = np.where(
            rolling_atr_mean != 0, 
            result_df[atr_col] / rolling_atr_mean, 
            1.0 # Default to 1 if mean is zero
        )
        result_df["volatility_ratio"].fillna(1.0, inplace=True) # Fill potential NaNs at the start
        
        # Detect regime
        result_df["market_regime"] = None
        result_df["regime_confidence"] = 0.0
        
        for i in range(lookback, len(result_df)):
            # Extract current metrics
            direction = result_df.iloc[i]["direction_strength"]
            vol_ratio = result_df.iloc[i]["volatility_ratio"]
            price_dist = result_df.iloc[i]["price_distance"]
            volume_surge = result_df.iloc[i]["volume_ratio"]
            price_change = result_df.iloc[i]["price_pct_change"]
            
            # Handle potential NaN values
            if pd.isna(direction) or pd.isna(vol_ratio) or pd.isna(price_dist) or pd.isna(volume_surge) or pd.isna(price_change):
                regime = self.MarketRegime.UNDEFINED
                confidence = 0.1
            else:
                # Determine regime
                if direction > 0.5:
                    # Strong directional movement
                    if price_change > 0:
                        regime = self.MarketRegime.TRENDING_UP
                    else:
                        regime = self.MarketRegime.TRENDING_DOWN
                    confidence = min(0.9, 0.5 + direction)
                elif vol_ratio < 0.7:
                    # Low volatility - ranging market
                    regime = self.MarketRegime.RANGING
                    confidence = min(0.9, 1.2 - vol_ratio)
                elif vol_ratio > 1.5:
                    # High volatility
                    if volume_surge > 2.0:
                        # High volume with high volatility - breakout
                        if price_change > 0:
                            regime = self.MarketRegime.BREAKOUT_UP
                        else:
                            regime = self.MarketRegime.BREAKOUT_DOWN
                        confidence = min(0.9, 0.5 + volume_surge / 3)
                    else:
                        # High volatility without volume confirmation - exhaustion
                        regime = self.MarketRegime.EXHAUSTION
                        confidence = min(0.8, 0.5 + vol_ratio / 3)
                elif vol_ratio > 1.0:
                    # Increasing volatility - expansion
                    regime = self.MarketRegime.EXPANSION
                    confidence = min(0.7, 0.4 + vol_ratio / 2)
                elif vol_ratio < 0.9:
                    # Decreasing volatility - contraction
                    regime = self.MarketRegime.CONTRACTION
                    confidence = min(0.7, 1.1 - vol_ratio)
                else:
                    # Undefined regime
                    regime = self.MarketRegime.UNDEFINED
                    confidence = 0.3
            
            # Update DataFrame
            result_df.iloc[i, result_df.columns.get_loc("market_regime")] = regime.value
            result_df.iloc[i, result_df.columns.get_loc("regime_confidence")] = confidence
        
        # Clean up intermediate columns
        cols_to_drop = [
            "up_move", "down_move", "smoothed_up", "smoothed_down",
            "price_change", "price_pct_change", "rolling_mean", "rolling_std",
            "rolling_volume_mean", "volume_ratio", "price_distance"
        ]
        result_df = result_df.drop(columns=[col for col in cols_to_drop if col in result_df.columns])
        
        return result_df
    
    def initialize_incremental(self) -> Dict[str, Any]:
        """Initialize state for incremental calculation"""
        state = {
            "price_buffer": [],
            "volume_buffer": [],
            "high_buffer": [],
            "low_buffer": [],
            "atr_buffer": [], # Buffer for pre-calculated ATR values
            "current_regime": self.MarketRegime.UNDEFINED.value,
            "regime_confidence": 0.3,
            "lookback_period": self.parameters["lookback_period"],
            "atr_period": self.parameters["atr_period"]
        }
        return state
    
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update market regime detection with new data incrementally (assumes ATR is pre-calculated)
        
        Args:
            state: Current calculation state
            new_data: New data point (must include pre-calculated ATR, e.g., 'ATR_14')
            
        Returns:
            Updated state with market regime
        """
        atr_col = f"ATR_{state['atr_period']}"
        if atr_col not in new_data:
             print(f"Warning: ATR column '{atr_col}' missing in new_data for incremental update.")
             return state # Cannot proceed without ATR

        # Add new data to buffers
        state["price_buffer"].append(new_data["close"])
        state["volume_buffer"].append(new_data["volume"])
        state["high_buffer"].append(new_data["high"])
        state["low_buffer"].append(new_data["low"])
        state["atr_buffer"].append(new_data[atr_col]) # Add ATR to buffer
        
        # Keep buffers at appropriate size
        lookback = state["lookback_period"]
        max_buffer_size = lookback * 2 # Keep enough history for rolling calculations
        if len(state["price_buffer"]) > max_buffer_size:
            state["price_buffer"] = state["price_buffer"][-max_buffer_size:]
            state["volume_buffer"] = state["volume_buffer"][-max_buffer_size:]
            state["high_buffer"] = state["high_buffer"][-max_buffer_size:]
            state["low_buffer"] = state["low_buffer"][-max_buffer_size:]
            state["atr_buffer"] = state["atr_buffer"][-max_buffer_size:]
        
        # Only proceed if we have enough data
        if len(state["price_buffer"]) < lookback:
            return state
        
        # Calculate rolling metrics
        rolling_mean = np.mean(state["price_buffer"][-lookback:])
        rolling_std = np.std(state["price_buffer"][-lookback:])
        rolling_volume_mean = np.mean(state["volume_buffer"][-lookback:])
        
        # Current metrics
        current_price = state["price_buffer"][-1]
        prev_price = state["price_buffer"][-2] if len(state["price_buffer"]) > 1 else current_price
        current_volume = state["volume_buffer"][-1]
        
        # Calculate price changes
        price_change = current_price - prev_price
        price_pct_change = price_change / prev_price if prev_price != 0 else 0
        
        # Volatility metrics
        if len(state["atr_buffer"]) >= lookback * 2:
            atr_mean = np.mean(state["atr_buffer"][-lookback*2:])
            volatility_ratio = new_data[atr_col] / atr_mean if atr_mean != 0 else 1.0
        else:
            volatility_ratio = 1.0 # Not enough history yet
        
        # Volume metrics
        volume_ratio = current_volume / rolling_volume_mean if rolling_volume_mean != 0 else 1.0
        
        # Price distance from mean
        price_distance = (current_price - rolling_mean) / rolling_std if rolling_std != 0 else 0
        
        # Directional strength calculation (simplified version)
        up_moves = []
        down_moves = []
        
        for i in range(1, min(lookback, len(state["high_buffer"]))):
            up_move = max(state["high_buffer"][-i] - state["high_buffer"][-i-1], 0)
            down_move = max(state["low_buffer"][-i-1] - state["low_buffer"][-i], 0)
            
            # Filter for significant moves only
            atr_threshold = 0.2
            if up_move > new_data[atr_col] * atr_threshold:
                up_moves.append(up_move)
            else:
                up_moves.append(0)
                
            if down_move > new_data[atr_col] * atr_threshold:
                down_moves.append(down_move)
            else:
                down_moves.append(0)
        
        smoothed_up = sum(up_moves)
        smoothed_down = sum(down_moves)
        direction_strength = abs(smoothed_up - smoothed_down) / (smoothed_up + smoothed_down) if (smoothed_up + smoothed_down) != 0 else 0
        
        # Determine regime
        regime = self.MarketRegime.UNDEFINED
        confidence = 0.3  # Base confidence
        
        if direction_strength > 0.5:
            # Strong directional movement
            if price_change > 0:
                regime = self.MarketRegime.TRENDING_UP
            else:
                regime = self.MarketRegime.TRENDING_DOWN
            confidence = min(0.9, 0.5 + direction_strength)
        elif volatility_ratio < 0.7:
            # Low volatility - ranging market
            regime = self.MarketRegime.RANGING
            confidence = min(0.9, 1.2 - volatility_ratio)
        elif volatility_ratio > 1.5:
            # High volatility
            if volume_ratio > 2.0:
                # High volume with high volatility - breakout
                if price_change > 0:
                    regime = self.MarketRegime.BREAKOUT_UP
                else:
                    regime = self.MarketRegime.BREAKOUT_DOWN
                confidence = min(0.9, 0.5 + volume_ratio / 3)
            else:
                # High volatility without volume confirmation - exhaustion
                regime = self.MarketRegime.EXHAUSTION
                confidence = min(0.8, 0.5 + volatility_ratio / 3)
        elif volatility_ratio > 1.0:
            # Increasing volatility - expansion
            regime = self.MarketRegime.EXPANSION
            confidence = min(0.7, 0.4 + volatility_ratio / 2)
        elif volatility_ratio < 0.9:
            # Decreasing volatility - contraction
            regime = self.MarketRegime.CONTRACTION
            confidence = min(0.7, 1.1 - volatility_ratio)
        
        # Update state with metrics
        state["direction_strength"] = direction_strength
        state["volatility_ratio"] = volatility_ratio
        state["volume_ratio"] = volume_ratio
        state["price_distance"] = price_distance
        state["price_pct_change"] = price_pct_change
        
        # Check for regime change threshold
        threshold = self.parameters["regime_change_threshold"]
        
        # Update regime based on confidence
        if confidence >= threshold:
            state["current_regime"] = regime.value
            state["regime_confidence"] = confidence
        elif confidence > 0.5 and state["regime_confidence"] < 0.8:
            blend_factor = confidence / (confidence + state["regime_confidence"])
            if blend_factor > 0.6:
                state["current_regime"] = regime.value
                state["regime_confidence"] = confidence
        # If confidence is low, potentially revert to Undefined or keep current
        elif confidence < 0.4 and state["regime_confidence"] > 0.5:
             # Don't immediately switch if confidence drops low
             pass
        elif confidence < 0.4:
             state["current_regime"] = self.MarketRegime.UNDEFINED.value
             state["regime_confidence"] = confidence

        return state


class VolumeVolatilityAnalyzer(CoreAdvancedAnalysisBase):
    """
    Analyzer for volume and volatility patterns
    
    Features:
    - Volume profile analysis
    - Volatility analysis
    - Volume-weighted price levels
    - Confluence detection with volume profile
    """
    
    def __init__(
        self,
        name: str = "VolumeVolatilityAnalyzer",
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the analyzer
        
        Args:
            name: Name of the analyzer
            parameters: Optional configuration parameters
        """
        super().__init__(name=name, parameters=parameters)
        self.confluence_analyzer = ConfluenceAnalyzer()
        
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze volume and volatility patterns
        
        Args:
            data: Dictionary containing market data and parameters
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Extract data
            df = pd.DataFrame(data["market_data"])
            
            # Calculate volume profile
            volume_profile = self._calculate_volume_profile(df)
            
            # Calculate volatility metrics
            volatility = self._calculate_volatility_metrics(df)
            
            # Run confluence analysis
            confluence_results = await self.confluence_analyzer.analyze(data)
            
            # Combine results
            result = {
                "volume_profile": volume_profile,
                "volatility": volatility,
                "confluence": confluence_results.result,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in volume volatility analysis: {str(e)}", exc_info=True)
            return {
                "error": f"Analysis failed: {str(e)}"
            }
            
    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume profile metrics"""
        # Implementation details...
        pass
        
    def _calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility metrics"""
        # Implementation details...
        pass
