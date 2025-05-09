"""
Feature Generation Operations

This module provides operations for generating features from market data.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from ..base_transformer import BaseMarketDataTransformer

logger = logging.getLogger(__name__)


class FeatureGenerator(BaseMarketDataTransformer):
    """
    Transformer for generating features from market data.
    
    This transformer generates technical indicators, statistical features,
    and other derived features from market data.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the feature generator.
        
        Args:
            parameters: Configuration parameters for the generator
        """
        default_params = {
            "moving_averages": [5, 10, 20, 50, 200],
            "rsi_periods": 14,
            "macd_params": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger_params": {"window": 20, "num_std": 2},
            "atr_periods": 14,
            "generate_all": True
        }
        
        # Merge default parameters with provided parameters
        merged_params = {**default_params, **(parameters or {})}
        super().__init__("feature_generator", merged_params)
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate features from market data.
        
        Args:
            data: Market data DataFrame
            **kwargs: Additional arguments for feature generation
            
        Returns:
            DataFrame with generated features
        """
        # Create a copy to avoid modifying the original
        transformed = data.copy()
        
        # Check if we have the necessary price data
        if 'close' not in transformed.columns:
            self.logger.warning("Close price not available, skipping feature generation")
            return transformed
        
        # Generate moving averages
        self._generate_moving_averages(transformed)
        
        # Generate momentum indicators
        self._generate_momentum_indicators(transformed)
        
        # Generate volatility indicators
        self._generate_volatility_indicators(transformed)
        
        # Generate volume indicators
        if 'volume' in transformed.columns:
            self._generate_volume_indicators(transformed)
        
        # Generate pattern recognition features
        if self.parameters.get("generate_patterns", False):
            self._generate_pattern_features(transformed)
        
        return transformed
    
    def get_required_columns(self) -> List[str]:
        """
        Get the list of required columns for this transformer.
        
        Returns:
            List of required column names
        """
        # Minimum required columns for feature generation
        return ["close"]
    
    def _generate_moving_averages(self, data: pd.DataFrame):
        """
        Generate moving average features.
        
        Args:
            data: Market data DataFrame
        """
        # Simple moving averages
        for period in self.parameters["moving_averages"]:
            data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
        
        # Exponential moving averages
        for period in self.parameters["moving_averages"]:
            data[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
        
        # Moving average crossovers
        if len(self.parameters["moving_averages"]) >= 2:
            # Sort periods
            periods = sorted(self.parameters["moving_averages"])
            
            # Generate crossover signals
            for i in range(len(periods) - 1):
                fast_period = periods[i]
                slow_period = periods[i + 1]
                
                # Crossover indicator (1 for golden cross, -1 for death cross, 0 for no cross)
                fast_ma = data[f'sma_{fast_period}']
                slow_ma = data[f'sma_{slow_period}']
                
                # Current state (fast above slow = 1, fast below slow = -1)
                data[f'ma_position_{fast_period}_{slow_period}'] = np.where(
                    fast_ma > slow_ma, 1, -1
                )
                
                # Crossover detection
                data[f'ma_crossover_{fast_period}_{slow_period}'] = np.where(
                    (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1)), 1,  # Golden cross
                    np.where(
                        (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1)), -1,  # Death cross
                        0  # No cross
                    )
                )
    
    def _generate_momentum_indicators(self, data: pd.DataFrame):
        """
        Generate momentum indicator features.
        
        Args:
            data: Market data DataFrame
        """
        # Relative Strength Index (RSI)
        rsi_periods = self.parameters["rsi_periods"]
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_periods).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_periods).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        macd_params = self.parameters["macd_params"]
        fast_ema = data['close'].ewm(span=macd_params["fast"], adjust=False).mean()
        slow_ema = data['close'].ewm(span=macd_params["slow"], adjust=False).mean()
        data['macd'] = fast_ema - slow_ema
        data['macd_signal'] = data['macd'].ewm(span=macd_params["signal"], adjust=False).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Rate of Change (ROC)
        data['roc_5'] = data['close'].pct_change(periods=5) * 100
        data['roc_10'] = data['close'].pct_change(periods=10) * 100
        data['roc_20'] = data['close'].pct_change(periods=20) * 100
    
    def _generate_volatility_indicators(self, data: pd.DataFrame):
        """
        Generate volatility indicator features.
        
        Args:
            data: Market data DataFrame
        """
        # Bollinger Bands
        bollinger_params = self.parameters["bollinger_params"]
        window = bollinger_params["window"]
        num_std = bollinger_params["num_std"]
        
        data['bb_middle'] = data['close'].rolling(window=window).mean()
        data['bb_std'] = data['close'].rolling(window=window).std()
        data['bb_upper'] = data['bb_middle'] + num_std * data['bb_std']
        data['bb_lower'] = data['bb_middle'] - num_std * data['bb_std']
        
        # Bollinger Band width
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # Bollinger Band position
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Average True Range (ATR)
        if all(col in data.columns for col in ['high', 'low', 'close']):
            atr_periods = self.parameters["atr_periods"]
            tr1 = data['high'] - data['low']
            tr2 = abs(data['high'] - data['close'].shift(1))
            tr3 = abs(data['low'] - data['close'].shift(1))
            data['true_range'] = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            data['atr'] = data['true_range'].rolling(window=atr_periods).mean()
            
            # ATR percentage
            data['atr_pct'] = data['atr'] / data['close'] * 100
    
    def _generate_volume_indicators(self, data: pd.DataFrame):
        """
        Generate volume indicator features.
        
        Args:
            data: Market data DataFrame
        """
        # Volume moving averages
        data['volume_sma_5'] = data['volume'].rolling(window=5).mean()
        data['volume_sma_10'] = data['volume'].rolling(window=10).mean()
        data['volume_sma_20'] = data['volume'].rolling(window=20).mean()
        
        # Relative volume
        data['relative_volume_5'] = data['volume'] / data['volume_sma_5']
        data['relative_volume_10'] = data['volume'] / data['volume_sma_10']
        data['relative_volume_20'] = data['volume'] / data['volume_sma_20']
        
        # On-Balance Volume (OBV)
        data['obv'] = (data['volume'] * np.where(data['close'] > data['close'].shift(1), 1,
                                              np.where(data['close'] < data['close'].shift(1), -1, 0))).cumsum()
        
        # Volume-weighted price
        if 'close' in data.columns:
            data['vwap'] = (data['volume'] * data['close']).cumsum() / data['volume'].cumsum()
    
    def _generate_pattern_features(self, data: pd.DataFrame):
        """
        Generate pattern recognition features.
        
        Args:
            data: Market data DataFrame
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated pattern recognition
        
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # Doji pattern (open and close are very close)
            data['doji'] = abs(data['close'] - data['open']) <= (data['high'] - data['low']) * 0.1
            
            # Hammer pattern (long lower shadow, small body, little or no upper shadow)
            data['hammer'] = (
                (data['high'] - data['low'] > 3 * abs(data['close'] - data['open'])) &  # Long range
                (((data['close'] > data['open']) &  # Bullish
                  (data['high'] - data['close'] < 0.3 * (data['high'] - data['low'])) &  # Small upper shadow
                  (data['open'] - data['low'] > 0.6 * (data['high'] - data['low']))) |  # Long lower shadow
                 ((data['open'] > data['close']) &  # Bearish
                  (data['high'] - data['open'] < 0.3 * (data['high'] - data['low'])) &  # Small upper shadow
                  (data['close'] - data['low'] > 0.6 * (data['high'] - data['low']))))  # Long lower shadow
            )
            
            # Engulfing pattern
            data['bullish_engulfing'] = (
                (data['close'].shift(1) < data['open'].shift(1)) &  # Previous candle is bearish
                (data['close'] > data['open']) &  # Current candle is bullish
                (data['close'] > data['open'].shift(1)) &  # Current close > previous open
                (data['open'] < data['close'].shift(1))  # Current open < previous close
            )
            
            data['bearish_engulfing'] = (
                (data['close'].shift(1) > data['open'].shift(1)) &  # Previous candle is bullish
                (data['close'] < data['open']) &  # Current candle is bearish
                (data['close'] < data['open'].shift(1)) &  # Current close < previous open
                (data['open'] > data['close'].shift(1))  # Current open > previous close
            )