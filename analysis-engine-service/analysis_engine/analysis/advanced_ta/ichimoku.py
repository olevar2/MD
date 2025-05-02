"""
Ichimoku Kinko Hyo (Ichimoku Cloud) Module

This module provides implementation of the Ichimoku Kinko Hyo (Ichimoku Cloud),
a comprehensive indicator system that provides information on potential support/resistance,
trend direction, momentum, and trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import math

from analysis_engine.analysis.advanced_ta.base import AdvancedAnalysisBase


class IchimokuCloud(AdvancedAnalysisBase):
    """
    Ichimoku Kinko Hyo (Ichimoku Cloud)
    
    Calculates all five components of the Ichimoku Cloud:
    - Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past tenkan_period
    - Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past kijun_period
    - Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 shifted forward by displacement
    - Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for past senkou_period, shifted forward
    - Chikou Span (Lagging Span): Current closing price shifted backwards by displacement
    """
    
    def __init__(
        self,
        name: str = "IchimokuCloud",
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize Ichimoku Cloud analyzer
        
        Args:
            name: Name of the analyzer
            parameters: Dictionary of parameters for analysis
        """
        default_params = {
            "tenkan_period": 9,      # Conversion Line period (short-term)
            "kijun_period": 26,      # Base Line period (medium-term)
            "senkou_period": 52,     # Leading Span B period (long-term)
            "displacement": 26,      # Displacement period for Senkou Span A/B and Chikou Span
            "include_chikou": True,  # Whether to include Chikou Span calculation
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name, default_params)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud components
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Ichimoku Cloud components
        """
        result_df = df.copy()
        
        # Extract parameters
        tenkan_period = self.parameters["tenkan_period"]
        kijun_period = self.parameters["kijun_period"]
        senkou_period = self.parameters["senkou_period"]
        displacement = self.parameters["displacement"]
        include_chikou = self.parameters["include_chikou"]
        
        # Calculate Tenkan-sen (Conversion Line)
        result_df['ichimoku_tenkan_sen'] = self._calculate_midpoint(
            result_df, tenkan_period
        )
        
        # Calculate Kijun-sen (Base Line)
        result_df['ichimoku_kijun_sen'] = self._calculate_midpoint(
            result_df, kijun_period
        )
        
        # Calculate Senkou Span A (Leading Span A)
        result_df['ichimoku_senkou_span_a'] = (
            (result_df['ichimoku_tenkan_sen'] + result_df['ichimoku_kijun_sen']) / 2
        ).shift(displacement)
        
        # Calculate Senkou Span B (Leading Span B)
        result_df['ichimoku_senkou_span_b'] = self._calculate_midpoint(
            result_df, senkou_period
        ).shift(displacement)
        
        # Calculate Chikou Span (Lagging Span)
        if include_chikou:
            result_df['ichimoku_chikou_span'] = result_df['close'].shift(-displacement)
        
        # Add cloud coloring indicator
        result_df['ichimoku_cloud_green'] = (
            result_df['ichimoku_senkou_span_a'] > result_df['ichimoku_senkou_span_b']
        ).astype(int)
        
        # Add cloud top and bottom for easy visualization
        result_df['ichimoku_cloud_top'] = result_df[['ichimoku_senkou_span_a', 'ichimoku_senkou_span_b']].max(axis=1)
        result_df['ichimoku_cloud_bottom'] = result_df[['ichimoku_senkou_span_a', 'ichimoku_senkou_span_b']].min(axis=1)
        
        # Add signal columns (basic Ichimoku signals)
        self._add_ichimoku_signals(result_df)
        
        return result_df
    
    def _calculate_midpoint(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate midpoint of highest high and lowest low over the given period
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for calculation
            
        Returns:
            Series with midpoint values
        """
        # Rolling window calculations
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        
        # Midpoint calculation
        return (high_max + low_min) / 2
    
    def _add_ichimoku_signals(self, df: pd.DataFrame) -> None:
        """
        Add common Ichimoku trading signals to the DataFrame
        
        Args:
            df: DataFrame with Ichimoku components
            
        Returns:
            None, modifies DataFrame in-place
        """
        # TK Cross (Tenkan-sen crosses Kijun-sen)
        df['ichimoku_tk_cross_bullish'] = (
            (df['ichimoku_tenkan_sen'] > df['ichimoku_kijun_sen']) & 
            (df['ichimoku_tenkan_sen'].shift(1) <= df['ichimoku_kijun_sen'].shift(1))
        ).astype(int)
        
        df['ichimoku_tk_cross_bearish'] = (
            (df['ichimoku_tenkan_sen'] < df['ichimoku_kijun_sen']) & 
            (df['ichimoku_tenkan_sen'].shift(1) >= df['ichimoku_kijun_sen'].shift(1))
        ).astype(int)
        
        # Price crosses Kijun-sen
        df['ichimoku_price_kijun_cross_bullish'] = (
            (df['close'] > df['ichimoku_kijun_sen']) & 
            (df['close'].shift(1) <= df['ichimoku_kijun_sen'].shift(1))
        ).astype(int)
        
        df['ichimoku_price_kijun_cross_bearish'] = (
            (df['close'] < df['ichimoku_kijun_sen']) & 
            (df['close'].shift(1) >= df['ichimoku_kijun_sen'].shift(1))
        ).astype(int)
        
        # Kumo breakout (Price crosses above/below the cloud)
        df['ichimoku_kumo_breakout_bullish'] = (
            (df['close'] > df['ichimoku_cloud_top']) &
            (df['close'].shift(1) <= df['ichimoku_cloud_top'].shift(1))
        ).astype(int)
        
        df['ichimoku_kumo_breakout_bearish'] = (
            (df['close'] < df['ichimoku_cloud_bottom']) &
            (df['close'].shift(1) >= df['ichimoku_cloud_bottom'].shift(1))
        ).astype(int)
        
        # Determine trend based on position relative to the cloud
        df['ichimoku_trend'] = np.where(
            df['close'] > df['ichimoku_cloud_top'], 1,  # Bullish trend
            np.where(df['close'] < df['ichimoku_cloud_bottom'], -1,  # Bearish trend
                     0)  # Inside cloud (no clear trend)
        )
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information"""
        return {
            'name': 'Ichimoku Cloud',
            'description': 'Comprehensive indicator showing support/resistance, trend direction, and momentum',
            'category': 'trend',
            'parameters': [
                {
                    'name': 'tenkan_period',
                    'description': 'Period for Tenkan-sen (Conversion Line)',
                    'type': 'int',
                    'default': 9
                },
                {
                    'name': 'kijun_period',
                    'description': 'Period for Kijun-sen (Base Line)',
                    'type': 'int',
                    'default': 26
                },
                {
                    'name': 'senkou_period',
                    'description': 'Period for Senkou Span B',
                    'type': 'int',
                    'default': 52
                },
                {
                    'name': 'displacement',
                    'description': 'Displacement period for cloud and Chikou Span',
                    'type': 'int',
                    'default': 26
                },
                {
                    'name': 'include_chikou',
                    'description': 'Include Chikou Span calculation',
                    'type': 'bool',
                    'default': True
                }
            ]
        }


class IchimokuKinkohyoWithAdditional(IchimokuCloud):
    """
    Extended Ichimoku Kinko Hyo with additional analysis
    
    Extends the standard Ichimoku Cloud with additional calculations, including:
    - Midpoint of the Kumo (cloud)
    - Senkou spans for different timeframes (fast, medium, slow)
    - Additional signals based on Chikou Span
    """
    
    def __init__(
        self,
        name: str = "IchimokuKinkohyoWithAdditional",
        parameters: Dict[str, Any] = None
    ):
        """Initialize Extended Ichimoku Cloud analyzer"""
        default_params = {
            "tenkan_period": 9,
            "kijun_period": 26,
            "senkou_period": 52,
            "displacement": 26,
            "include_chikou": True,
            "include_kumo_midpoint": True,
            "include_multi_timeframe": False,
            "include_additional_signals": True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name, default_params)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Extended Ichimoku Cloud components
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with extended Ichimoku Cloud components
        """
        # First, calculate standard Ichimoku Cloud components
        result_df = super().calculate(df)
        
        # Calculate Kumo midpoint if requested
        if self.parameters["include_kumo_midpoint"]:
            result_df['ichimoku_kumo_midpoint'] = (
                (result_df['ichimoku_senkou_span_a'] + result_df['ichimoku_senkou_span_b']) / 2
            )
        
        # Calculate multi-timeframe Senkou spans if requested
        if self.parameters["include_multi_timeframe"]:
            # Fast cloud (shorter periods)
            fast_tenkan = max(3, self.parameters["tenkan_period"] // 2)
            fast_kijun = max(6, self.parameters["kijun_period"] // 2)
            fast_senkou = max(13, self.parameters["senkou_period"] // 2)
            
            # Calculate fast components
            tenkan_fast = self._calculate_midpoint(result_df, fast_tenkan)
            kijun_fast = self._calculate_midpoint(result_df, fast_kijun)
            
            result_df['ichimoku_senkou_span_a_fast'] = (
                (tenkan_fast + kijun_fast) / 2
            ).shift(self.parameters["displacement"])
            
            result_df['ichimoku_senkou_span_b_fast'] = self._calculate_midpoint(
                result_df, fast_senkou
            ).shift(self.parameters["displacement"])
            
            # Slow cloud (longer periods)
            slow_tenkan = self.parameters["tenkan_period"] * 2
            slow_kijun = self.parameters["kijun_period"] * 2
            slow_senkou = self.parameters["senkou_period"] * 2
            
            # Calculate slow components
            tenkan_slow = self._calculate_midpoint(result_df, slow_tenkan)
            kijun_slow = self._calculate_midpoint(result_df, slow_kijun)
            
            result_df['ichimoku_senkou_span_a_slow'] = (
                (tenkan_slow + kijun_slow) / 2
            ).shift(self.parameters["displacement"])
            
            result_df['ichimoku_senkou_span_b_slow'] = self._calculate_midpoint(
                result_df, slow_senkou
            ).shift(self.parameters["displacement"])
        
        # Add additional signals if requested
        if self.parameters["include_additional_signals"]:
            # Add Chikou Span signals if Chikou is included
            if self.parameters["include_chikou"]:
                # Chikou Span crosses price
                result_df['ichimoku_chikou_price_cross_bullish'] = (
                    (result_df['ichimoku_chikou_span'] > result_df['close']) & 
                    (result_df['ichimoku_chikou_span'].shift(1) <= result_df['close'].shift(1))
                ).astype(int)
                
                result_df['ichimoku_chikou_price_cross_bearish'] = (
                    (result_df['ichimoku_chikou_span'] < result_df['close']) & 
                    (result_df['ichimoku_chikou_span'].shift(1) >= result_df['close'].shift(1))
                ).astype(int)
                
                # Chikou position relative to cloud
                result_df['ichimoku_chikou_above_cloud'] = (
                    result_df['ichimoku_chikou_span'] > result_df['ichimoku_cloud_top']
                ).astype(int)
                
                result_df['ichimoku_chikou_below_cloud'] = (
                    result_df['ichimoku_chikou_span'] < result_df['ichimoku_cloud_bottom']
                ).astype(int)
            
            # Add flat Kumo detection (when the cloud is flat, market often in consolidation)
            result_df['ichimoku_kumo_flat'] = (
                abs(result_df['ichimoku_senkou_span_a'] - result_df['ichimoku_senkou_span_b']) < 
                (result_df['ichimoku_senkou_span_a'] * 0.01)  # Within 1% of each other
            ).astype(int)
            
            # Add strong trend indicator (price, Tenkan, Kijun all aligned)
            result_df['ichimoku_strong_bullish_trend'] = (
                (result_df['close'] > result_df['ichimoku_tenkan_sen']) &
                (result_df['ichimoku_tenkan_sen'] > result_df['ichimoku_kijun_sen']) &
                (result_df['ichimoku_kijun_sen'] > result_df['ichimoku_cloud_top'])
            ).astype(int)
            
            result_df['ichimoku_strong_bearish_trend'] = (
                (result_df['close'] < result_df['ichimoku_tenkan_sen']) &
                (result_df['ichimoku_tenkan_sen'] < result_df['ichimoku_kijun_sen']) &
                (result_df['ichimoku_kijun_sen'] < result_df['ichimoku_cloud_bottom'])
            ).astype(int)
        
        return result_df
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information"""
        return {
            'name': 'Extended Ichimoku Kinko Hyo',
            'description': 'Ichimoku Cloud with additional analysis and signals',
            'category': 'trend',
            'parameters': [
                {
                    'name': 'tenkan_period',
                    'description': 'Period for Tenkan-sen (Conversion Line)',
                    'type': 'int',
                    'default': 9
                },
                {
                    'name': 'kijun_period',
                    'description': 'Period for Kijun-sen (Base Line)',
                    'type': 'int',
                    'default': 26
                },
                {
                    'name': 'senkou_period',
                    'description': 'Period for Senkou Span B',
                    'type': 'int',
                    'default': 52
                },
                {
                    'name': 'displacement',
                    'description': 'Displacement period for cloud and Chikou Span',
                    'type': 'int',
                    'default': 26
                },
                {
                    'name': 'include_chikou',
                    'description': 'Include Chikou Span calculation',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'include_kumo_midpoint',
                    'description': 'Include Kumo midpoint calculation',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'include_multi_timeframe',
                    'description': 'Include multi-timeframe analysis',
                    'type': 'bool',
                    'default': False
                },
                {
                    'name': 'include_additional_signals',
                    'description': 'Include additional trading signals',
                    'type': 'bool',
                    'default': True
                }
            ]
        }
