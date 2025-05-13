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
from analysis_engine.analysis.advanced_ta.base import AdvancedAnalysisBase, ConfidenceLevel, MarketDirection, AnalysisTimeframe
from analysis_engine.core.base.components import AdvancedAnalysisBase as CoreAdvancedAnalysisBase
from analysis_engine.analysis.confluence.confluence_analyzer import ConfluenceAnalyzer
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class MarketRegimeAnalyzer(AdvancedAnalysisBase):
    """
    Market Regime Detector
    
    Uses volatility and volume metrics (assumed pre-calculated) to detect market regimes.
    """


    class MarketRegime(Enum):
    """
    MarketRegime class that inherits from Enum.
    
    Attributes:
        Add attributes here
    """

        TRENDING_UP = 'Trending Up'
        TRENDING_DOWN = 'Trending Down'
        RANGING = 'Ranging'
        VOLATILE = 'Volatile'
        BREAKOUT_UP = 'Breakout Up'
        BREAKOUT_DOWN = 'Breakout Down'
        EXHAUSTION = 'Exhaustion'
        EXPANSION = 'Expansion'
        CONTRACTION = 'Contraction'
        UNDEFINED = 'Undefined'

    def init(self, lookback_period: int=20, atr_period: int=14,
        regime_change_threshold: float=0.7):
        """
        Initialize the Market Regime Detector
        
        Args:
            lookback_period: Period for calculating market metrics
            atr_period: Period for referencing the pre-calculated ATR column (e.g., 'ATR_14')
            regime_change_threshold: Threshold for regime change confirmation
        """
        parameters = {'lookback_period': lookback_period, 'atr_period':
            atr_period, 'regime_change_threshold': regime_change_threshold}
        super().__init__('Market Regime Detector', parameters)

    def calculate(self, df: pd.DataFrame) ->pd.DataFrame:
        """
        Detect market regimes for the given data (assumes ATR is pre-calculated)
        
        Args:
            df: DataFrame with OHLCV data and pre-calculated ATR (e.g., 'ATR_14')
            
        Returns:
            DataFrame with market regime labels and metrics
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        atr_col = f"ATR_{self.parameters['atr_period']}"
        required_cols.append(atr_col)
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(
                f'DataFrame must contain columns: {required_cols}. Missing: {missing}'
                )
        result_df = df.copy()
        result_df['price_change'] = result_df['close'].diff()
        result_df['price_pct_change'] = result_df['close'].pct_change()
        lookback = self.parameters['lookback_period']
        result_df['rolling_mean'] = result_df['close'].rolling(window=lookback
            ).mean()
        result_df['rolling_std'] = result_df['close'].rolling(window=lookback
            ).std()
        result_df['rolling_volume_mean'] = result_df['volume'].rolling(window
            =lookback).mean()
        result_df['volume_ratio'] = result_df['volume'] / result_df[
            'rolling_volume_mean']
        result_df['price_distance'] = (result_df['close'] - result_df[
            'rolling_mean']) / result_df['rolling_std']
        result_df['up_move'] = np.maximum(result_df['high'] - result_df[
            'high'].shift(1), 0)
        result_df['down_move'] = np.maximum(result_df['low'].shift(1) -
            result_df['low'], 0)
        atr_threshold = 0.2
        result_df['up_move'] = np.where(result_df['up_move'] > result_df[
            atr_col] * atr_threshold, result_df['up_move'], 0)
        result_df['down_move'] = np.where(result_df['down_move'] > 
            result_df[atr_col] * atr_threshold, result_df['down_move'], 0)
        result_df['smoothed_up'] = result_df['up_move'].rolling(window=lookback
            ).sum()
        result_df['smoothed_down'] = result_df['down_move'].rolling(window=
            lookback).sum()
        denominator = result_df['smoothed_up'] + result_df['smoothed_down']
        result_df['direction_strength'] = np.where(denominator != 0, abs(
            result_df['smoothed_up'] - result_df['smoothed_down']) /
            denominator, 0)
        result_df['direction_strength'].fillna(0, inplace=True)
        rolling_atr_mean = result_df[atr_col].rolling(window=lookback * 2
            ).mean()
        result_df['volatility_ratio'] = np.where(rolling_atr_mean != 0, 
            result_df[atr_col] / rolling_atr_mean, 1.0)
        result_df['volatility_ratio'].fillna(1.0, inplace=True)
        result_df['market_regime'] = None
        result_df['regime_confidence'] = 0.0
        for i in range(lookback, len(result_df)):
            direction = result_df.iloc[i]['direction_strength']
            vol_ratio = result_df.iloc[i]['volatility_ratio']
            price_dist = result_df.iloc[i]['price_distance']
            volume_surge = result_df.iloc[i]['volume_ratio']
            price_change = result_df.iloc[i]['price_pct_change']
            if pd.isna(direction) or pd.isna(vol_ratio) or pd.isna(price_dist
                ) or pd.isna(volume_surge) or pd.isna(price_change):
                regime = self.MarketRegime.UNDEFINED
                confidence = 0.1
            elif direction > 0.5:
                if price_change > 0:
                    regime = self.MarketRegime.TRENDING_UP
                else:
                    regime = self.MarketRegime.TRENDING_DOWN
                confidence = min(0.9, 0.5 + direction)
            elif vol_ratio < 0.7:
                regime = self.MarketRegime.RANGING
                confidence = min(0.9, 1.2 - vol_ratio)
            elif vol_ratio > 1.5:
                if volume_surge > 2.0:
                    if price_change > 0:
                        regime = self.MarketRegime.BREAKOUT_UP
                    else:
                        regime = self.MarketRegime.BREAKOUT_DOWN
                    confidence = min(0.9, 0.5 + volume_surge / 3)
                else:
                    regime = self.MarketRegime.EXHAUSTION
                    confidence = min(0.8, 0.5 + vol_ratio / 3)
            elif vol_ratio > 1.0:
                regime = self.MarketRegime.EXPANSION
                confidence = min(0.7, 0.4 + vol_ratio / 2)
            elif vol_ratio < 0.9:
                regime = self.MarketRegime.CONTRACTION
                confidence = min(0.7, 1.1 - vol_ratio)
            else:
                regime = self.MarketRegime.UNDEFINED
                confidence = 0.3
            result_df.iloc[i, result_df.columns.get_loc('market_regime')
                ] = regime.value
            result_df.iloc[i, result_df.columns.get_loc('regime_confidence')
                ] = confidence
        cols_to_drop = ['up_move', 'down_move', 'smoothed_up',
            'smoothed_down', 'price_change', 'price_pct_change',
            'rolling_mean', 'rolling_std', 'rolling_volume_mean',
            'volume_ratio', 'price_distance']
        result_df = result_df.drop(columns=[col for col in cols_to_drop if 
            col in result_df.columns])
        return result_df

    def initialize_incremental(self) ->Dict[str, Any]:
        """Initialize state for incremental calculation"""
        state = {'price_buffer': [], 'volume_buffer': [], 'high_buffer': [],
            'low_buffer': [], 'atr_buffer': [], 'current_regime': self.
            MarketRegime.UNDEFINED.value, 'regime_confidence': 0.3,
            'lookback_period': self.parameters['lookback_period'],
            'atr_period': self.parameters['atr_period']}
        return state

    @with_resilience('update_incremental')
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str,
        Any]) ->Dict[str, Any]:
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
            print(
                f"Warning: ATR column '{atr_col}' missing in new_data for incremental update."
                )
            return state
        state['price_buffer'].append(new_data['close'])
        state['volume_buffer'].append(new_data['volume'])
        state['high_buffer'].append(new_data['high'])
        state['low_buffer'].append(new_data['low'])
        state['atr_buffer'].append(new_data[atr_col])
        lookback = state['lookback_period']
        max_buffer_size = lookback * 2
        if len(state['price_buffer']) > max_buffer_size:
            state['price_buffer'] = state['price_buffer'][-max_buffer_size:]
            state['volume_buffer'] = state['volume_buffer'][-max_buffer_size:]
            state['high_buffer'] = state['high_buffer'][-max_buffer_size:]
            state['low_buffer'] = state['low_buffer'][-max_buffer_size:]
            state['atr_buffer'] = state['atr_buffer'][-max_buffer_size:]
        if len(state['price_buffer']) < lookback:
            return state
        rolling_mean = np.mean(state['price_buffer'][-lookback:])
        rolling_std = np.std(state['price_buffer'][-lookback:])
        rolling_volume_mean = np.mean(state['volume_buffer'][-lookback:])
        current_price = state['price_buffer'][-1]
        prev_price = state['price_buffer'][-2] if len(state['price_buffer']
            ) > 1 else current_price
        current_volume = state['volume_buffer'][-1]
        price_change = current_price - prev_price
        price_pct_change = price_change / prev_price if prev_price != 0 else 0
        if len(state['atr_buffer']) >= lookback * 2:
            atr_mean = np.mean(state['atr_buffer'][-lookback * 2:])
            volatility_ratio = new_data[atr_col
                ] / atr_mean if atr_mean != 0 else 1.0
        else:
            volatility_ratio = 1.0
        volume_ratio = (current_volume / rolling_volume_mean if 
            rolling_volume_mean != 0 else 1.0)
        price_distance = (current_price - rolling_mean
            ) / rolling_std if rolling_std != 0 else 0
        up_moves = []
        down_moves = []
        for i in range(1, min(lookback, len(state['high_buffer']))):
            up_move = max(state['high_buffer'][-i] - state['high_buffer'][-
                i - 1], 0)
            down_move = max(state['low_buffer'][-i - 1] - state[
                'low_buffer'][-i], 0)
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
        direction_strength = abs(smoothed_up - smoothed_down) / (smoothed_up +
            smoothed_down) if smoothed_up + smoothed_down != 0 else 0
        regime = self.MarketRegime.UNDEFINED
        confidence = 0.3
        if direction_strength > 0.5:
            if price_change > 0:
                regime = self.MarketRegime.TRENDING_UP
            else:
                regime = self.MarketRegime.TRENDING_DOWN
            confidence = min(0.9, 0.5 + direction_strength)
        elif volatility_ratio < 0.7:
            regime = self.MarketRegime.RANGING
            confidence = min(0.9, 1.2 - volatility_ratio)
        elif volatility_ratio > 1.5:
            if volume_ratio > 2.0:
                if price_change > 0:
                    regime = self.MarketRegime.BREAKOUT_UP
                else:
                    regime = self.MarketRegime.BREAKOUT_DOWN
                confidence = min(0.9, 0.5 + volume_ratio / 3)
            else:
                regime = self.MarketRegime.EXHAUSTION
                confidence = min(0.8, 0.5 + volatility_ratio / 3)
        elif volatility_ratio > 1.0:
            regime = self.MarketRegime.EXPANSION
            confidence = min(0.7, 0.4 + volatility_ratio / 2)
        elif volatility_ratio < 0.9:
            regime = self.MarketRegime.CONTRACTION
            confidence = min(0.7, 1.1 - volatility_ratio)
        state['direction_strength'] = direction_strength
        state['volatility_ratio'] = volatility_ratio
        state['volume_ratio'] = volume_ratio
        state['price_distance'] = price_distance
        state['price_pct_change'] = price_pct_change
        threshold = self.parameters['regime_change_threshold']
        if confidence >= threshold:
            state['current_regime'] = regime.value
            state['regime_confidence'] = confidence
        elif confidence > 0.5 and state['regime_confidence'] < 0.8:
            blend_factor = confidence / (confidence + state[
                'regime_confidence'])
            if blend_factor > 0.6:
                state['current_regime'] = regime.value
                state['regime_confidence'] = confidence
        elif confidence < 0.4 and state['regime_confidence'] > 0.5:
            pass
        elif confidence < 0.4:
            state['current_regime'] = self.MarketRegime.UNDEFINED.value
            state['regime_confidence'] = confidence
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

    def __init__(self, name: str='VolumeVolatilityAnalyzer', parameters:
        Optional[Dict[str, Any]]=None):
        """
        Initialize the analyzer
        
        Args:
            name: Name of the analyzer
            parameters: Optional configuration parameters
        """
        super().__init__(name=name, parameters=parameters)
        self.confluence_analyzer = ConfluenceAnalyzer()

    @async_with_exception_handling
    async def analyze(self, data: Dict[str, Any]) ->Dict[str, Any]:
        """
        Analyze volume and volatility patterns
        
        Args:
            data: Dictionary containing market data and parameters
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            df = pd.DataFrame(data['market_data'])
            volume_profile = self._calculate_volume_profile(df)
            volatility = self._calculate_volatility_metrics(df)
            confluence_results = await self.confluence_analyzer.analyze(data)
            result = {'volume_profile': volume_profile, 'volatility':
                volatility, 'confluence': confluence_results.result,
                'timestamp': datetime.now().isoformat()}
            return result
        except Exception as e:
            logger.error(f'Error in volume volatility analysis: {str(e)}',
                exc_info=True)
            return {'error': f'Analysis failed: {str(e)}'}

    def _calculate_volume_profile(self, df: pd.DataFrame) ->Dict[str, Any]:
        """Calculate volume profile metrics"""
        pass

    def _calculate_volatility_metrics(self, df: pd.DataFrame) ->Dict[str, Any]:
        """Calculate volatility metrics"""
        pass
