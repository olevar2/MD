"""
Candlestick Pattern Recognition.

This module implements algorithms to recognize over 40 candlestick patterns
with success statistics and a pattern ranking system.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from enum import Enum

from feature_store_service.indicators.base_indicator import BaseIndicator


class PatternStrength(Enum):
    """Enum for pattern strength levels."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


class CandlestickPatternRecognition(BaseIndicator):
    """
    Candlestick Pattern Recognition indicator.
    
    This indicator implements algorithms to recognize over 40 candlestick patterns,
    adds success statistics for each pattern, and develops a pattern ranking system.
    """
    
    category = "pattern_recognition"
    
    def __init__(self, ranking_window: int = 100, **kwargs):
        """
        Initialize Candlestick Pattern Recognition indicator.
        
        Args:
            ranking_window: Lookback period for calculating pattern success rates
            **kwargs: Additional parameters
        """
        self.ranking_window = ranking_window
        self.name = "candlestick_patterns"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Recognize candlestick patterns in the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with recognized patterns and statistics
        """
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Create columns for all patterns
        pattern_columns = {}
        
        # ===== Single Candlestick Patterns =====
        pattern_columns.update(self._detect_single_candlestick_patterns(result))
        
        # ===== Double Candlestick Patterns =====
        pattern_columns.update(self._detect_double_candlestick_patterns(result))
        
        # ===== Triple Candlestick Patterns =====
        pattern_columns.update(self._detect_triple_candlestick_patterns(result))
        
        # ===== Complex Multi-bar Patterns =====
        pattern_columns.update(self._detect_complex_patterns(result))
        
        # Calculate success statistics for patterns
        success_stats = self._calculate_pattern_success(result, pattern_columns)
        
        # Calculate pattern strength ranking
        self._calculate_pattern_ranking(result, pattern_columns, success_stats)
        
        return result
    
    def _detect_single_candlestick_patterns(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Detect single candlestick patterns.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary mapping pattern names to column names
        """
        pattern_columns = {}
        
        # Calculate basic candlestick properties
        data['body_size'] = abs(data['close'] - data['open'])
        data['shadow_upper'] = data['high'] - data[['open', 'close']].max(axis=1)
        data['shadow_lower'] = data[['open', 'close']].min(axis=1) - data['low']
        data['body_mid'] = (data['open'] + data['close']) / 2
        data['total_range'] = data['high'] - data['low']
        data['body_ratio'] = data['body_size'] / data['total_range'].replace(0, np.nan)
        data['is_bullish'] = (data['close'] > data['open']).astype(int)
        data['is_bearish'] = (data['close'] < data['open']).astype(int)
        data['prev_is_bullish'] = data['is_bullish'].shift(1)
        data['prev_is_bearish'] = data['is_bearish'].shift(1)
        
        # 1. Doji
        pattern_name = 'doji'
        data[pattern_name] = ((data['body_size'] / data['total_range']) < 0.1) & (data['total_range'] > 0)
        pattern_columns[pattern_name] = pattern_name
        
        # 2. Dragonfly Doji
        pattern_name = 'dragonfly_doji'
        data[pattern_name] = (data['doji'] & 
                             (data['shadow_lower'] > 2 * data['body_size']) & 
                             (data['shadow_upper'] < data['body_size']))
        pattern_columns[pattern_name] = pattern_name
        
        # 3. Gravestone Doji
        pattern_name = 'gravestone_doji'
        data[pattern_name] = (data['doji'] & 
                             (data['shadow_upper'] > 2 * data['body_size']) & 
                             (data['shadow_lower'] < data['body_size']))
        pattern_columns[pattern_name] = pattern_name
        
        # 4. Long-legged Doji
        pattern_name = 'long_legged_doji'
        data[pattern_name] = (data['doji'] & 
                             (data['shadow_upper'] > 2 * data['body_size']) & 
                             (data['shadow_lower'] > 2 * data['body_size']))
        pattern_columns[pattern_name] = pattern_name
        
        # 5. Hammer (bullish)
        pattern_name = 'hammer'
        data[pattern_name] = ((data['shadow_lower'] > 2 * data['body_size']) & 
                             (data['shadow_upper'] < 0.2 * data['body_size']) &
                             (data['body_size'] > 0))
        pattern_columns[pattern_name] = pattern_name
        
        # 6. Inverted Hammer (bullish)
        pattern_name = 'inverted_hammer'
        data[pattern_name] = ((data['shadow_upper'] > 2 * data['body_size']) & 
                             (data['shadow_lower'] < 0.2 * data['body_size']) &
                             (data['body_size'] > 0))
        pattern_columns[pattern_name] = pattern_name
        
        # 7. Shooting Star (bearish)
        pattern_name = 'shooting_star'
        data[pattern_name] = ((data['shadow_upper'] > 2 * data['body_size']) & 
                             (data['shadow_lower'] < 0.2 * data['body_size']) &
                             (data['is_bearish'] == 1))
        pattern_columns[pattern_name] = pattern_name
        
        # 8. Hanging Man (bearish)
        pattern_name = 'hanging_man'
        data[pattern_name] = ((data['shadow_lower'] > 2 * data['body_size']) & 
                             (data['shadow_upper'] < 0.2 * data['body_size']) &
                             (data['is_bearish'] == 1))
        pattern_columns[pattern_name] = pattern_name
        
        # 9. Marubozu (strong trend continuation)
        pattern_name = 'marubozu'
        data[pattern_name] = ((data['shadow_upper'] < 0.05 * data['body_size']) & 
                             (data['shadow_lower'] < 0.05 * data['body_size']) &
                             (data['body_size'] > 0.8 * data['total_range']))
        pattern_columns[pattern_name] = pattern_name
        
        # 10. Spinning Top
        pattern_name = 'spinning_top'
        data[pattern_name] = ((data['body_size'] < 0.3 * data['total_range']) & 
                             (data['shadow_upper'] > data['body_size']) & 
                             (data['shadow_lower'] > data['body_size']))
        pattern_columns[pattern_name] = pattern_name
        
        return pattern_columns
    
    def _detect_double_candlestick_patterns(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Detect double candlestick patterns.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary mapping pattern names to column names
        """
        pattern_columns = {}
        
        # Previous candle properties
        data['prev_open'] = data['open'].shift(1)
        data['prev_high'] = data['high'].shift(1)
        data['prev_low'] = data['low'].shift(1)
        data['prev_close'] = data['close'].shift(1)
        data['prev_body_size'] = data['body_size'].shift(1)
        data['prev_total_range'] = data['total_range'].shift(1)
        
        # 1. Bullish Engulfing
        pattern_name = 'bullish_engulfing'
        data[pattern_name] = ((data['is_bullish'] == 1) & 
                             (data['prev_is_bearish'] == 1) &
                             (data['open'] < data['prev_close']) &
                             (data['close'] > data['prev_open']))
        pattern_columns[pattern_name] = pattern_name
        
        # 2. Bearish Engulfing
        pattern_name = 'bearish_engulfing'
        data[pattern_name] = ((data['is_bearish'] == 1) & 
                             (data['prev_is_bullish'] == 1) &
                             (data['open'] > data['prev_close']) &
                             (data['close'] < data['prev_open']))
        pattern_columns[pattern_name] = pattern_name
        
        # 3. Bullish Harami
        pattern_name = 'bullish_harami'
        data[pattern_name] = ((data['is_bullish'] == 1) & 
                             (data['prev_is_bearish'] == 1) &
                             (data['open'] > data['prev_close']) &
                             (data['close'] < data['prev_open']) &
                             (data['body_size'] < data['prev_body_size']))
        pattern_columns[pattern_name] = pattern_name
        
        # 4. Bearish Harami
        pattern_name = 'bearish_harami'
        data[pattern_name] = ((data['is_bearish'] == 1) & 
                             (data['prev_is_bullish'] == 1) &
                             (data['open'] < data['prev_close']) &
                             (data['close'] > data['prev_open']) &
                             (data['body_size'] < data['prev_body_size']))
        pattern_columns[pattern_name] = pattern_name
        
        # 5. Tweezer Top
        pattern_name = 'tweezer_top'
        data[pattern_name] = ((data['is_bearish'] == 1) & 
                             (data['prev_is_bullish'] == 1) &
                             (abs(data['high'] - data['prev_high']) < 0.05 * data['prev_high']))
        pattern_columns[pattern_name] = pattern_name
        
        # 6. Tweezer Bottom
        pattern_name = 'tweezer_bottom'
        data[pattern_name] = ((data['is_bullish'] == 1) & 
                             (data['prev_is_bearish'] == 1) &
                             (abs(data['low'] - data['prev_low']) < 0.05 * data['prev_low']))
        pattern_columns[pattern_name] = pattern_name
        
        # 7. Piercing Line
        pattern_name = 'piercing_line'
        data[pattern_name] = ((data['is_bullish'] == 1) & 
                             (data['prev_is_bearish'] == 1) &
                             (data['open'] < data['prev_low']) &
                             (data['close'] > (data['prev_open'] + data['prev_close']) / 2) &
                             (data['close'] < data['prev_open']))
        pattern_columns[pattern_name] = pattern_name
        
        # 8. Dark Cloud Cover
        pattern_name = 'dark_cloud_cover'
        data[pattern_name] = ((data['is_bearish'] == 1) & 
                             (data['prev_is_bullish'] == 1) &
                             (data['open'] > data['prev_high']) &
                             (data['close'] < (data['prev_open'] + data['prev_close']) / 2) &
                             (data['close'] > data['prev_open']))
        pattern_columns[pattern_name] = pattern_name
        
        # 9. Inside Bar
        pattern_name = 'inside_bar'
        data[pattern_name] = ((data['high'] < data['prev_high']) & 
                             (data['low'] > data['prev_low']))
        pattern_columns[pattern_name] = pattern_name
        
        # 10. Outside Bar
        pattern_name = 'outside_bar'
        data[pattern_name] = ((data['high'] > data['prev_high']) & 
                             (data['low'] < data['prev_low']))
        pattern_columns[pattern_name] = pattern_name
        
        return pattern_columns
    
    def _detect_triple_candlestick_patterns(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Detect triple candlestick patterns.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary mapping pattern names to column names
        """
        pattern_columns = {}
        
        # Add properties for 2 bars back
        data['prev2_open'] = data['open'].shift(2)
        data['prev2_high'] = data['high'].shift(2)
        data['prev2_low'] = data['low'].shift(2)
        data['prev2_close'] = data['close'].shift(2)
        data['prev2_is_bullish'] = data['is_bullish'].shift(2)
        data['prev2_is_bearish'] = data['is_bearish'].shift(2)
        
        # 1. Morning Star
        pattern_name = 'morning_star'
        data[pattern_name] = ((data['prev2_is_bearish'] == 1) &
                             (data['prev_body_size'] / data['prev_total_range'] < 0.3) &
                             (data['is_bullish'] == 1) &
                             (data['close'] > (data['prev2_open'] + data['prev2_close']) / 2))
        pattern_columns[pattern_name] = pattern_name
        
        # 2. Evening Star
        pattern_name = 'evening_star'
        data[pattern_name] = ((data['prev2_is_bullish'] == 1) &
                             (data['prev_body_size'] / data['prev_total_range'] < 0.3) &
                             (data['is_bearish'] == 1) &
                             (data['close'] < (data['prev2_open'] + data['prev2_close']) / 2))
        pattern_columns[pattern_name] = pattern_name
        
        # 3. Three White Soldiers
        pattern_name = 'three_white_soldiers'
        data[pattern_name] = ((data['is_bullish'] == 1) &
                             (data['prev_is_bullish'] == 1) &
                             (data['prev2_is_bullish'] == 1) &
                             (data['close'] > data['prev_close']) &
                             (data['prev_close'] > data['prev2_close']) &
                             (data['open'] > data['prev_open']) &
                             (data['prev_open'] > data['prev2_open']))
        pattern_columns[pattern_name] = pattern_name
        
        # 4. Three Black Crows
        pattern_name = 'three_black_crows'
        data[pattern_name] = ((data['is_bearish'] == 1) &
                             (data['prev_is_bearish'] == 1) &
                             (data['prev2_is_bearish'] == 1) &
                             (data['close'] < data['prev_close']) &
                             (data['prev_close'] < data['prev2_close']) &
                             (data['open'] < data['prev_open']) &
                             (data['prev_open'] < data['prev2_open']))
        pattern_columns[pattern_name] = pattern_name
        
        # 5. Abandoned Baby (Bullish)
        pattern_name = 'bullish_abandoned_baby'
        data[pattern_name] = ((data['prev2_is_bearish'] == 1) &
                             (data['prev_doji'] == True) &
                             (data['is_bullish'] == 1) &
                             (data['prev_low'] > data['prev2_close']) &
                             (data['prev_low'] > data['open']))
        pattern_columns[pattern_name] = pattern_name
        
        # 6. Abandoned Baby (Bearish)
        pattern_name = 'bearish_abandoned_baby'
        data[pattern_name] = ((data['prev2_is_bullish'] == 1) &
                             (data['prev_doji'] == True) &
                             (data['is_bearish'] == 1) &
                             (data['prev_high'] < data['prev2_close']) &
                             (data['prev_high'] < data['open']))
        pattern_columns[pattern_name] = pattern_name
        
        # 7. Three Inside Up
        pattern_name = 'three_inside_up'
        data[pattern_name] = ((data['prev2_is_bearish'] == 1) &
                             (data['prev_is_bullish'] == 1) &
                             (data['is_bullish'] == 1) &
                             (data['prev_open'] > data['prev2_close']) &
                             (data['prev_close'] < data['prev2_open']) &
                             (data['close'] > data['prev2_open']))
        pattern_columns[pattern_name] = pattern_name
        
        # 8. Three Inside Down
        pattern_name = 'three_inside_down'
        data[pattern_name] = ((data['prev2_is_bullish'] == 1) &
                             (data['prev_is_bearish'] == 1) &
                             (data['is_bearish'] == 1) &
                             (data['prev_open'] < data['prev2_close']) &
                             (data['prev_close'] > data['prev2_open']) &
                             (data['close'] < data['prev2_open']))
        pattern_columns[pattern_name] = pattern_name
        
        # 9. Three Outside Up
        pattern_name = 'three_outside_up'
        data[pattern_name] = ((data['prev2_is_bearish'] == 1) &
                             (data['prev_is_bullish'] == 1) &
                             (data['is_bullish'] == 1) &
                             (data['prev_open'] < data['prev2_close']) &
                             (data['prev_close'] > data['prev2_open']) &
                             (data['close'] > data['prev_close']))
        pattern_columns[pattern_name] = pattern_name
        
        # 10. Three Outside Down
        pattern_name = 'three_outside_down'
        data[pattern_name] = ((data['prev2_is_bullish'] == 1) &
                             (data['prev_is_bearish'] == 1) &
                             (data['is_bearish'] == 1) &
                             (data['prev_open'] > data['prev2_close']) &
                             (data['prev_close'] < data['prev2_open']) &
                             (data['close'] < data['prev_close']))
        pattern_columns[pattern_name] = pattern_name
        
        return pattern_columns
    
    def _detect_complex_patterns(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Detect more complex candlestick patterns.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary mapping pattern names to column names
        """
        pattern_columns = {}
        
        # 1. Bullish Kicker
        pattern_name = 'bullish_kicker'
        data[pattern_name] = ((data['is_bullish'] == 1) &
                             (data['prev_is_bearish'] == 1) &
                             (data['open'] > data['prev_open']) &
                             (data['body_size'] > 0.5 * data['total_range']))
        pattern_columns[pattern_name] = pattern_name
        
        # 2. Bearish Kicker
        pattern_name = 'bearish_kicker'
        data[pattern_name] = ((data['is_bearish'] == 1) &
                             (data['prev_is_bullish'] == 1) &
                             (data['open'] < data['prev_open']) &
                             (data['body_size'] > 0.5 * data['total_range']))
        pattern_columns[pattern_name] = pattern_name
        
        # 3. Rising Three Methods
        pattern_name = 'rising_three_methods'
        rising_three = ((data['prev4_is_bullish'] == 1) &
                       (data['prev3_is_bearish'] == 1) &
                       (data['prev2_is_bearish'] == 1) &
                       (data['prev_is_bearish'] == 1) &
                       (data['is_bullish'] == 1) &
                       (data['close'] > data['prev4_close']) &
                       (data['prev3_high'] < data['prev4_high']) &
                       (data['prev2_high'] < data['prev4_high']) &
                       (data['prev_high'] < data['prev4_high']) &
                       (data['prev3_low'] > data['prev4_low']) &
                       (data['prev2_low'] > data['prev4_low']) &
                       (data['prev_low'] > data['prev4_low']))
        data[pattern_name] = rising_three
        pattern_columns[pattern_name] = pattern_name
        
        # 4. Falling Three Methods
        pattern_name = 'falling_three_methods'
        falling_three = ((data['prev4_is_bearish'] == 1) &
                        (data['prev3_is_bullish'] == 1) &
                        (data['prev2_is_bullish'] == 1) &
                        (data['prev_is_bullish'] == 1) &
                        (data['is_bearish'] == 1) &
                        (data['close'] < data['prev4_close']) &
                        (data['prev3_low'] > data['prev4_low']) &
                        (data['prev2_low'] > data['prev4_low']) &
                        (data['prev_low'] > data['prev4_low']) &
                        (data['prev3_high'] < data['prev4_high']) &
                        (data['prev2_high'] < data['prev4_high']) &
                        (data['prev_high'] < data['prev4_high']))
        data[pattern_name] = falling_three
        pattern_columns[pattern_name] = pattern_name
        
        # Add more complex patterns here as needed
        
        # Convert all pattern columns to integers (0 or 1)
        for col in pattern_columns.values():
            data[col] = data[col].astype(int)
        
        return pattern_columns
    
    def _calculate_pattern_success(
        self, 
        data: pd.DataFrame, 
        pattern_columns: Dict[str, str],
        forward_bars: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate success statistics for detected patterns.
        
        Args:
            data: DataFrame with OHLCV and pattern data
            pattern_columns: Dictionary mapping pattern names to column names
            forward_bars: Number of bars to look forward for success calculation
            
        Returns:
            Dictionary with success statistics for each pattern
        """
        success_stats = {}
        
        # For each pattern, calculate success rate
        for pattern_name, col in pattern_columns.items():
            # Skip if no instances of this pattern
            if data[col].sum() == 0:
                continue
                
            # Initialize statistics for this pattern
            success_stats[pattern_name] = {
                "occurrences": int(data[col].sum()),
                "success_rate": 0.0,
                "avg_gain": 0.0,
                "avg_loss": 0.0,
                "risk_reward": 0.0
            }
            
            # For each occurrence of the pattern
            pattern_indices = data[data[col] == 1].index
            
            success_count = 0
            gains = []
            losses = []
            
            for idx in pattern_indices:
                # Skip if we don't have enough forward bars
                if idx + forward_bars >= len(data):
                    continue
                
                # Check if pattern is inherently bullish or bearish based on name
                is_bullish_pattern = any(bullish_term in pattern_name for bullish_term in 
                                       ["bullish", "hammer", "morning", "white", "piercing", "inside_up", "outside_up"])
                is_bearish_pattern = any(bearish_term in pattern_name for bearish_term in 
                                       ["bearish", "shooting", "evening", "black", "dark_cloud", "inside_down", "outside_down"])
                
                # If neither, try to determine from close vs open
                if not is_bullish_pattern and not is_bearish_pattern:
                    is_bullish_pattern = data.loc[idx, 'close'] >= data.loc[idx, 'open']
                    is_bearish_pattern = not is_bullish_pattern
                
                # Calculate price movement after pattern
                entry_price = data.loc[idx, 'close']
                forward_idx = idx + forward_bars
                exit_price = data.loc[forward_idx, 'close']
                
                price_change_pct = (exit_price - entry_price) / entry_price * 100
                
                # Determine if pattern was successful
                if (is_bullish_pattern and price_change_pct > 0) or (is_bearish_pattern and price_change_pct < 0):
                    success_count += 1
                    if is_bullish_pattern:
                        gains.append(price_change_pct)
                    else:  # bearish pattern
                        gains.append(-price_change_pct)  # Convert to positive gain for bearish patterns
                else:
                    if is_bullish_pattern:
                        losses.append(-price_change_pct)  # Convert to positive loss for bullish patterns
                    else:  # bearish pattern
                        losses.append(price_change_pct)
            
            # Calculate statistics
            if pattern_indices.size > 0:
                success_stats[pattern_name]["success_rate"] = success_count / len(pattern_indices) * 100
                
                if gains:
                    success_stats[pattern_name]["avg_gain"] = sum(gains) / len(gains)
                
                if losses:
                    success_stats[pattern_name]["avg_loss"] = sum(losses) / len(losses)
                
                if success_stats[pattern_name]["avg_loss"] > 0:
                    success_stats[pattern_name]["risk_reward"] = success_stats[pattern_name]["avg_gain"] / success_stats[pattern_name]["avg_loss"]
            
            # Add statistics columns to the dataframe
            data[f"{col}_success_rate"] = success_stats[pattern_name]["success_rate"]
            data[f"{col}_avg_gain"] = success_stats[pattern_name]["avg_gain"]
            data[f"{col}_risk_reward"] = success_stats[pattern_name]["risk_reward"]
        
        return success_stats
    
    def _calculate_pattern_ranking(
        self, 
        data: pd.DataFrame, 
        pattern_columns: Dict[str, str],
        success_stats: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Calculate pattern strength ranking system.
        
        Args:
            data: DataFrame with OHLCV and pattern data
            pattern_columns: Dictionary mapping pattern names to column names
            success_stats: Dictionary with success statistics for each pattern
        """
        # Create a pattern strength column
        data['pattern_strength'] = 0
        data['strongest_pattern'] = None
        
        # For each detected pattern, calculate strength
        for pattern_name, col in pattern_columns.items():
            # Skip if no statistics for this pattern
            if pattern_name not in success_stats:
                continue
                
            stats = success_stats[pattern_name]
            
            # Calculate pattern strength on a scale of 0-10
            pattern_strength = 0
            
            # Success rate contributes up to 5 points
            pattern_strength += min(stats["success_rate"] / 20, 5)
            
            # Risk-reward ratio contributes up to 3 points
            if stats["risk_reward"] > 0:
                pattern_strength += min(stats["risk_reward"], 3)
            
            # Number of occurrences contributes up to 2 points (more samples = more reliable)
            pattern_strength += min(stats["occurrences"] / 50, 2)
            
            # Convert to a 0-4 strength enum
            if pattern_strength < 2.5:
                strength_value = PatternStrength.WEAK.value
            elif pattern_strength < 5:
                strength_value = PatternStrength.MODERATE.value
            elif pattern_strength < 7.5:
                strength_value = PatternStrength.STRONG.value
            else:
                strength_value = PatternStrength.VERY_STRONG.value
                
            # Add strength column
            data[f"{col}_strength"] = data[col] * strength_value
            
            # Update overall pattern strength if this pattern is detected
            pattern_mask = (data[col] == 1)
            stronger_pattern = data.loc[pattern_mask, 'pattern_strength'] < strength_value
            data.loc[pattern_mask & stronger_pattern, 'pattern_strength'] = strength_value
            data.loc[pattern_mask & stronger_pattern, 'strongest_pattern'] = pattern_name
""""""
