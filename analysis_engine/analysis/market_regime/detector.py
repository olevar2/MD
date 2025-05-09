"""
Market Regime Detector

This module provides functionality for detecting market regimes through
feature extraction from price data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from analysis_engine.analysis.market_regime.models import FeatureSet


class RegimeDetector:
    """
    Extracts features from price data to detect market regimes.
    
    This class is responsible for calculating technical indicators and
    extracting features that can be used to classify market regimes.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RegimeDetector.
        
        Args:
            config: Optional configuration dictionary with parameters:
                - lookback_periods: Dict mapping feature names to lookback periods
                - atr_period: Period for ATR calculation (default: 14)
                - adx_period: Period for ADX calculation (default: 14)
                - rsi_period: Period for RSI calculation (default: 14)
                - volatility_threshold: Dict with 'low', 'medium', 'high' thresholds
        """
        self.config = config or {}
        self.lookback_periods = self.config.get('lookback_periods', {
            'volatility': 14,
            'trend_strength': 14,
            'momentum': 14,
            'mean_reversion': 14,
            'range_width': 20
        })
        
        self.atr_period = self.config.get('atr_period', 14)
        self.adx_period = self.config.get('adx_period', 14)
        self.rsi_period = self.config.get('rsi_period', 14)
        
        self.volatility_threshold = self.config.get('volatility_threshold', {
            'low': 0.5,
            'medium': 1.0,
            'high': 2.0
        })
    
    def extract_features(self, price_data: pd.DataFrame) -> FeatureSet:
        """
        Extract features from price data for regime detection.
        
        Args:
            price_data: DataFrame with OHLCV data
                Required columns: 'open', 'high', 'low', 'close', 'volume'
                
        Returns:
            FeatureSet: A set of features for regime classification
        """
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in price_data.columns for col in required_columns):
            raise ValueError(f"Price data must contain columns: {required_columns}")
        
        # Calculate technical indicators
        atr = self._calculate_atr(price_data)
        adx = self._calculate_adx(price_data)
        rsi = self._calculate_rsi(price_data)
        
        # Extract features
        volatility = self._extract_volatility(price_data, atr)
        trend_strength = self._extract_trend_strength(price_data, adx)
        momentum = self._extract_momentum(price_data, rsi)
        mean_reversion = self._extract_mean_reversion(price_data)
        range_width = self._extract_range_width(price_data)
        
        # Additional features
        additional_features = {
            'price_velocity': self._calculate_price_velocity(price_data),
            'volume_trend': self._calculate_volume_trend(price_data) if 'volume' in price_data.columns else 0.0,
            'swing_strength': self._calculate_swing_strength(price_data)
        }
        
        return FeatureSet(
            volatility=volatility,
            trend_strength=trend_strength,
            momentum=momentum,
            mean_reversion=mean_reversion,
            range_width=range_width,
            additional_features=additional_features
        )
    
    def _calculate_atr(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()
        
        return atr
    
    def _calculate_adx(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate Average Directional Index."""
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']
        
        # Plus Directional Movement (+DM)
        plus_dm = high.diff()
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > -low.diff()), 0)
        
        # Minus Directional Movement (-DM)
        minus_dm = low.diff()
        minus_dm = minus_dm.where((minus_dm < 0) & (minus_dm < -high.diff()), 0)
        minus_dm = abs(minus_dm)
        
        # True Range
        tr = self._calculate_atr(price_data)
        
        # Smoothed +DM and -DM
        plus_di = 100 * (plus_dm.rolling(self.adx_period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(self.adx_period).mean() / tr)
        
        # Directional Index (DX)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Average Directional Index (ADX)
        adx = dx.rolling(self.adx_period).mean()
        
        return adx
    
    def _calculate_rsi(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = price_data['close'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _extract_volatility(self, price_data: pd.DataFrame, atr: pd.Series) -> float:
        """Extract volatility feature from price data."""
        # Normalize ATR by price level
        normalized_atr = atr / price_data['close']
        
        # Get the most recent value
        recent_volatility = normalized_atr.iloc[-1]
        
        # Compare to historical volatility
        historical_volatility = normalized_atr.rolling(
            self.lookback_periods['volatility']).mean().iloc[-1]
        
        # Return relative volatility (current vs historical)
        return float(recent_volatility / historical_volatility if historical_volatility > 0 else 1.0)
    
    def _extract_trend_strength(self, price_data: pd.DataFrame, adx: pd.Series) -> float:
        """Extract trend strength feature from price data."""
        # ADX above 25 indicates a strong trend
        # ADX below 20 indicates a weak trend
        recent_adx = adx.iloc[-1]
        
        # Normalize to 0-1 range
        normalized_trend_strength = min(recent_adx / 50.0, 1.0)
        
        return float(normalized_trend_strength)
    
    def _extract_momentum(self, price_data: pd.DataFrame, rsi: pd.Series) -> float:
        """Extract momentum feature from price data."""
        # RSI measures momentum
        recent_rsi = rsi.iloc[-1]
        
        # Normalize to -1 to 1 range (50 is neutral)
        normalized_momentum = (recent_rsi - 50) / 50.0
        
        return float(normalized_momentum)
    
    def _extract_mean_reversion(self, price_data: pd.DataFrame) -> float:
        """Extract mean reversion tendency from price data."""
        close = price_data['close']
        
        # Calculate distance from moving average
        ma_period = self.lookback_periods['mean_reversion']
        ma = close.rolling(ma_period).mean()
        
        # Distance from MA as percentage
        distance = (close - ma) / ma
        
        # Recent distance
        recent_distance = distance.iloc[-1]
        
        # Historical tendency to revert to mean
        # Correlation between distance and next period's return
        shifted_returns = close.pct_change().shift(-1)
        correlation = distance.iloc[:-1].corr(shifted_returns.iloc[:-1])
        
        # Combine recent distance and historical tendency
        mean_reversion = recent_distance * correlation
        
        return float(mean_reversion)
    
    def _extract_range_width(self, price_data: pd.DataFrame) -> float:
        """Extract range width feature from price data."""
        high = price_data['high']
        low = price_data['low']
        
        # Calculate recent range
        period = self.lookback_periods['range_width']
        recent_high = high.rolling(period).max().iloc[-1]
        recent_low = low.rolling(period).min().iloc[-1]
        
        # Range as percentage of current price
        range_width = (recent_high - recent_low) / price_data['close'].iloc[-1]
        
        return float(range_width)
    
    def _calculate_price_velocity(self, price_data: pd.DataFrame) -> float:
        """Calculate price velocity (rate of change over time)."""
        close = price_data['close']
        
        # Rate of change over different periods
        roc_5 = close.pct_change(5).iloc[-1]
        roc_10 = close.pct_change(10).iloc[-1]
        roc_20 = close.pct_change(20).iloc[-1]
        
        # Weighted average of different periods
        velocity = (0.5 * roc_5 + 0.3 * roc_10 + 0.2 * roc_20)
        
        return float(velocity)
    
    def _calculate_volume_trend(self, price_data: pd.DataFrame) -> float:
        """Calculate volume trend."""
        if 'volume' not in price_data.columns:
            return 0.0
            
        volume = price_data['volume']
        close = price_data['close']
        
        # Volume moving average
        vol_ma = volume.rolling(10).mean()
        
        # Volume ratio (current to average)
        vol_ratio = volume / vol_ma
        
        # Correlation between volume and price changes
        price_change = close.pct_change()
        vol_price_corr = vol_ratio.iloc[-10:].corr(price_change.iloc[-10:])
        
        # Recent volume trend
        recent_vol_trend = vol_ratio.iloc[-1] * vol_price_corr
        
        return float(recent_vol_trend)
    
    def _calculate_swing_strength(self, price_data: pd.DataFrame) -> float:
        """Calculate swing strength based on recent highs and lows."""
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']
        
        # Find recent swing points
        window = 5
        highs = high.rolling(window*2+1, center=True).max()
        lows = low.rolling(window*2+1, center=True).min()
        
        # Identify swing highs and lows
        swing_highs = high[(high == highs) & (high.shift(window) != high) & (high.shift(-window) != high)]
        swing_lows = low[(low == lows) & (low.shift(window) != low) & (low.shift(-window) != low)]
        
        # Get recent swings
        recent_swings = pd.concat([swing_highs, swing_lows]).sort_index().iloc[-4:]
        
        if len(recent_swings) < 2:
            return 0.0
            
        # Calculate average swing size
        swing_sizes = abs(recent_swings.diff())
        avg_swing = swing_sizes.mean()
        
        # Normalize by current price
        normalized_swing = avg_swing / close.iloc[-1]
        
        return float(normalized_swing)