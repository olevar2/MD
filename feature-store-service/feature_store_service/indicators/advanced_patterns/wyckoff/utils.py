"""
Wyckoff Pattern Utilities Module.

This module provides utility functions for Wyckoff pattern recognition.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from feature_store_service.indicators.advanced_patterns.wyckoff.models import (
    WyckoffPatternType,
    WyckoffPhase,
    WyckoffSchematic
)


def detect_accumulation_phase(
    data: pd.DataFrame,
    lookback: int = 100,
    volume_weight: float = 0.6,
    price_weight: float = 0.4,
    sensitivity: float = 0.75
) -> List[WyckoffSchematic]:
    """
    Detect Wyckoff accumulation phases in price data.
    
    Args:
        data: DataFrame with OHLCV data
        lookback: Number of bars to look back
        volume_weight: Weight of volume confirmation in pattern strength
        price_weight: Weight of price action in pattern strength
        sensitivity: Sensitivity of pattern detection (0.0-1.0)
        
    Returns:
        List of WyckoffSchematic objects representing accumulation patterns
    """
    patterns = []
    
    # Ensure we have enough data
    if len(data) < lookback:
        return patterns
    
    # Get the subset of data to analyze
    subset = data.iloc[-lookback:].copy()
    
    # Calculate price and volume metrics
    subset['price_range'] = subset['high'] - subset['low']
    subset['avg_volume'] = subset['volume'].rolling(window=20).mean()
    subset['volume_ratio'] = subset['volume'] / subset['avg_volume']
    
    # Identify potential selling climax (Phase A)
    # Selling climax is characterized by wide price range, high volume, and a close near the high
    subset['selling_climax'] = (
        (subset['price_range'] > subset['price_range'].rolling(window=20).mean() * 1.5) &
        (subset['volume'] > subset['avg_volume'] * 1.5) &
        (subset['close'] > (subset['low'] + subset['price_range'] * 0.7))
    ).astype(int)
    
    # Identify automatic rally after selling climax
    subset['automatic_rally'] = 0
    for i in range(5, len(subset)):
        if subset['selling_climax'].iloc[i-5:i].sum() > 0:
            # Look for a rally within 5 bars after selling climax
            if subset['close'].iloc[i] > subset['close'].iloc[i-5:i].max():
                subset.loc[subset.index[i], 'automatic_rally'] = 1
    
    # Identify secondary test (retest of lows)
    subset['secondary_test'] = 0
    for i in range(10, len(subset)):
        if subset['automatic_rally'].iloc[i-10:i].sum() > 0:
            # Look for a test of lows within 10 bars after automatic rally
            if (subset['low'].iloc[i] <= subset['low'].iloc[i-10:i].min() * 1.02) and \
               (subset['volume'].iloc[i] < subset['volume'].iloc[i-10:i].mean()):
                subset.loc[subset.index[i], 'secondary_test'] = 1
    
    # Identify spring (Phase C)
    subset['spring'] = 0
    for i in range(15, len(subset)):
        if subset['secondary_test'].iloc[i-15:i].sum() > 0:
            # Spring is a brief penetration of support with lower volume
            if (subset['low'].iloc[i] < subset['low'].iloc[i-15:i].min()) and \
               (subset['close'].iloc[i] > subset['low'].iloc[i-15:i].min()) and \
               (subset['volume'].iloc[i] < subset['volume'].iloc[i-15:i].mean()):
                subset.loc[subset.index[i], 'spring'] = 1
    
    # Identify sign of strength (Phase D)
    subset['sign_of_strength'] = 0
    for i in range(5, len(subset)):
        if subset['spring'].iloc[i-5:i].sum() > 0:
            # Sign of strength is a strong up move after spring
            if (subset['close'].iloc[i] > subset['high'].iloc[i-5:i].max()) and \
               (subset['volume'].iloc[i] > subset['avg_volume'].iloc[i]):
                subset.loc[subset.index[i], 'sign_of_strength'] = 1
    
    # Identify last point of support (Phase D)
    subset['last_point_of_support'] = 0
    for i in range(5, len(subset)):
        if subset['sign_of_strength'].iloc[i-5:i].sum() > 0:
            # Last point of support is a higher low after sign of strength
            if (subset['low'].iloc[i] > subset['low'].iloc[i-5:i].min()) and \
               (subset['low'].iloc[i] < subset['close'].iloc[i-5:i].mean()) and \
               (subset['volume'].iloc[i] < subset['avg_volume'].iloc[i]):
                subset.loc[subset.index[i], 'last_point_of_support'] = 1
    
    # Identify breakout (Phase E)
    subset['breakout'] = 0
    for i in range(5, len(subset)):
        if subset['last_point_of_support'].iloc[i-5:i].sum() > 0:
            # Breakout is a strong up move with high volume
            if (subset['close'].iloc[i] > subset['high'].iloc[i-5:i].max()) and \
               (subset['volume'].iloc[i] > subset['avg_volume'].iloc[i] * 1.5):
                subset.loc[subset.index[i], 'breakout'] = 1
    
    # Find complete accumulation patterns
    for i in range(30, len(subset)):
        # Check if we have a breakout
        if subset['breakout'].iloc[i] == 1:
            # Look back for the complete pattern
            window = subset.iloc[i-30:i+1]
            
            if window['selling_climax'].sum() > 0 and \
               window['automatic_rally'].sum() > 0 and \
               window['secondary_test'].sum() > 0 and \
               window['spring'].sum() > 0 and \
               window['sign_of_strength'].sum() > 0 and \
               window['last_point_of_support'].sum() > 0:
                
                # We have a complete accumulation pattern
                
                # Find the start index (selling climax)
                start_idx = window.index[window['selling_climax'] == 1][0]
                start_idx_pos = data.index.get_loc(start_idx)
                
                # Find the end index (breakout)
                end_idx = window.index[i]
                end_idx_pos = data.index.get_loc(end_idx)
                
                # Find phase transitions
                phase_a_start = start_idx_pos
                phase_a_end = data.index.get_loc(window.index[window['secondary_test'] == 1][0])
                
                phase_b_start = phase_a_end + 1
                phase_b_end = data.index.get_loc(window.index[window['spring'] == 1][0]) - 1
                
                phase_c_start = phase_b_end + 1
                phase_c_end = data.index.get_loc(window.index[window['spring'] == 1][0])
                
                phase_d_start = phase_c_end + 1
                phase_d_end = data.index.get_loc(window.index[window['last_point_of_support'] == 1][0])
                
                phase_e_start = phase_d_end + 1
                phase_e_end = end_idx_pos
                
                # Create phases dictionary
                phases = {
                    WyckoffPhase.PHASE_A_ACC: (phase_a_start, phase_a_end),
                    WyckoffPhase.PHASE_B_ACC: (phase_b_start, phase_b_end),
                    WyckoffPhase.PHASE_C_ACC: (phase_c_start, phase_c_end),
                    WyckoffPhase.PHASE_D_ACC: (phase_d_start, phase_d_end),
                    WyckoffPhase.PHASE_E_ACC: (phase_e_start, phase_e_end)
                }
                
                # Calculate pattern strength
                volume_confirmation = min(1.0, window['volume_ratio'].mean() / 2.0)
                price_confirmation = min(1.0, (window['close'].iloc[-1] - window['low'].min()) / 
                                        (window['high'].max() - window['low'].min()))
                
                strength = volume_confirmation * volume_weight + price_confirmation * price_weight
                strength = min(1.0, strength * sensitivity)
                
                # Calculate target price (measured move)
                pattern_height = window['high'].max() - window['low'].min()
                target_price = window['close'].iloc[-1] + pattern_height
                
                # Calculate stop price (below spring low)
                spring_low = window.loc[window['spring'] == 1, 'low'].min()
                stop_price = spring_low * 0.98  # 2% below spring low
                
                # Create the pattern
                pattern = WyckoffSchematic(
                    pattern_type=WyckoffPatternType.ACCUMULATION,
                    start_index=start_idx_pos,
                    end_index=end_idx_pos,
                    current_phase=WyckoffPhase.PHASE_E_ACC,
                    phases=phases,
                    direction="bullish",
                    strength=strength,
                    volume_confirms=volume_confirmation > 0.7,
                    target_price=target_price,
                    stop_price=stop_price
                )
                
                patterns.append(pattern)
    
    return patterns


def detect_distribution_phase(
    data: pd.DataFrame,
    lookback: int = 100,
    volume_weight: float = 0.6,
    price_weight: float = 0.4,
    sensitivity: float = 0.75
) -> List[WyckoffSchematic]:
    """
    Detect Wyckoff distribution phases in price data.
    
    Args:
        data: DataFrame with OHLCV data
        lookback: Number of bars to look back
        volume_weight: Weight of volume confirmation in pattern strength
        price_weight: Weight of price action in pattern strength
        sensitivity: Sensitivity of pattern detection (0.0-1.0)
        
    Returns:
        List of WyckoffSchematic objects representing distribution patterns
    """
    patterns = []
    
    # Ensure we have enough data
    if len(data) < lookback:
        return patterns
    
    # Get the subset of data to analyze
    subset = data.iloc[-lookback:].copy()
    
    # Calculate price and volume metrics
    subset['price_range'] = subset['high'] - subset['low']
    subset['avg_volume'] = subset['volume'].rolling(window=20).mean()
    subset['volume_ratio'] = subset['volume'] / subset['avg_volume']
    
    # Identify preliminary supply (Phase A)
    subset['preliminary_supply'] = (
        (subset['high'] > subset['high'].rolling(window=20).max()) &
        (subset['close'] < subset['high'] - subset['price_range'] * 0.3) &
        (subset['volume'] > subset['avg_volume'] * 1.2)
    ).astype(int)
    
    # Identify buying climax (Phase A)
    subset['buying_climax'] = (
        (subset['price_range'] > subset['price_range'].rolling(window=20).mean() * 1.5) &
        (subset['volume'] > subset['avg_volume'] * 1.5) &
        (subset['close'] < (subset['high'] - subset['price_range'] * 0.3))
    ).astype(int)
    
    # Identify automatic reaction after buying climax
    subset['automatic_reaction'] = 0
    for i in range(5, len(subset)):
        if subset['buying_climax'].iloc[i-5:i].sum() > 0:
            # Look for a reaction within 5 bars after buying climax
            if subset['close'].iloc[i] < subset['close'].iloc[i-5:i].min():
                subset.loc[subset.index[i], 'automatic_reaction'] = 1
    
    # Identify secondary test (retest of highs)
    subset['secondary_test'] = 0
    for i in range(10, len(subset)):
        if subset['automatic_reaction'].iloc[i-10:i].sum() > 0:
            # Look for a test of highs within 10 bars after automatic reaction
            if (subset['high'].iloc[i] >= subset['high'].iloc[i-10:i].max() * 0.98) and \
               (subset['volume'].iloc[i] < subset['volume'].iloc[i-10:i].mean()):
                subset.loc[subset.index[i], 'secondary_test'] = 1
    
    # Identify upthrust (Phase C)
    subset['upthrust'] = 0
    for i in range(15, len(subset)):
        if subset['secondary_test'].iloc[i-15:i].sum() > 0:
            # Upthrust is a brief penetration of resistance with lower volume
            if (subset['high'].iloc[i] > subset['high'].iloc[i-15:i].max()) and \
               (subset['close'].iloc[i] < subset['high'].iloc[i-15:i].max()) and \
               (subset['volume'].iloc[i] < subset['volume'].iloc[i-15:i].mean()):
                subset.loc[subset.index[i], 'upthrust'] = 1
    
    # Identify sign of weakness (Phase D)
    subset['sign_of_weakness'] = 0
    for i in range(5, len(subset)):
        if subset['upthrust'].iloc[i-5:i].sum() > 0:
            # Sign of weakness is a strong down move after upthrust
            if (subset['close'].iloc[i] < subset['low'].iloc[i-5:i].min()) and \
               (subset['volume'].iloc[i] > subset['avg_volume'].iloc[i]):
                subset.loc[subset.index[i], 'sign_of_weakness'] = 1
    
    # Identify last point of supply (Phase D)
    subset['last_point_of_supply'] = 0
    for i in range(5, len(subset)):
        if subset['sign_of_weakness'].iloc[i-5:i].sum() > 0:
            # Last point of supply is a lower high after sign of weakness
            if (subset['high'].iloc[i] < subset['high'].iloc[i-5:i].max()) and \
               (subset['high'].iloc[i] > subset['close'].iloc[i-5:i].mean()) and \
               (subset['volume'].iloc[i] < subset['avg_volume'].iloc[i]):
                subset.loc[subset.index[i], 'last_point_of_supply'] = 1
    
    # Identify breakdown (Phase E)
    subset['breakdown'] = 0
    for i in range(5, len(subset)):
        if subset['last_point_of_supply'].iloc[i-5:i].sum() > 0:
            # Breakdown is a strong down move with high volume
            if (subset['close'].iloc[i] < subset['low'].iloc[i-5:i].min()) and \
               (subset['volume'].iloc[i] > subset['avg_volume'].iloc[i] * 1.5):
                subset.loc[subset.index[i], 'breakdown'] = 1
    
    # Find complete distribution patterns
    for i in range(30, len(subset)):
        # Check if we have a breakdown
        if subset['breakdown'].iloc[i] == 1:
            # Look back for the complete pattern
            window = subset.iloc[i-30:i+1]
            
            if window['preliminary_supply'].sum() > 0 and \
               window['buying_climax'].sum() > 0 and \
               window['automatic_reaction'].sum() > 0 and \
               window['secondary_test'].sum() > 0 and \
               window['upthrust'].sum() > 0 and \
               window['sign_of_weakness'].sum() > 0 and \
               window['last_point_of_supply'].sum() > 0:
                
                # We have a complete distribution pattern
                
                # Find the start index (preliminary supply)
                start_idx = window.index[window['preliminary_supply'] == 1][0]
                start_idx_pos = data.index.get_loc(start_idx)
                
                # Find the end index (breakdown)
                end_idx = window.index[i]
                end_idx_pos = data.index.get_loc(end_idx)
                
                # Find phase transitions
                phase_a_start = start_idx_pos
                phase_a_end = data.index.get_loc(window.index[window['secondary_test'] == 1][0])
                
                phase_b_start = phase_a_end + 1
                phase_b_end = data.index.get_loc(window.index[window['upthrust'] == 1][0]) - 1
                
                phase_c_start = phase_b_end + 1
                phase_c_end = data.index.get_loc(window.index[window['upthrust'] == 1][0])
                
                phase_d_start = phase_c_end + 1
                phase_d_end = data.index.get_loc(window.index[window['last_point_of_supply'] == 1][0])
                
                phase_e_start = phase_d_end + 1
                phase_e_end = end_idx_pos
                
                # Create phases dictionary
                phases = {
                    WyckoffPhase.PHASE_A_DIST: (phase_a_start, phase_a_end),
                    WyckoffPhase.PHASE_B_DIST: (phase_b_start, phase_b_end),
                    WyckoffPhase.PHASE_C_DIST: (phase_c_start, phase_c_end),
                    WyckoffPhase.PHASE_D_DIST: (phase_d_start, phase_d_end),
                    WyckoffPhase.PHASE_E_DIST: (phase_e_start, phase_e_end)
                }
                
                # Calculate pattern strength
                volume_confirmation = min(1.0, window['volume_ratio'].mean() / 2.0)
                price_confirmation = min(1.0, (window['high'].max() - window['close'].iloc[-1]) / 
                                        (window['high'].max() - window['low'].min()))
                
                strength = volume_confirmation * volume_weight + price_confirmation * price_weight
                strength = min(1.0, strength * sensitivity)
                
                # Calculate target price (measured move)
                pattern_height = window['high'].max() - window['low'].min()
                target_price = window['close'].iloc[-1] - pattern_height
                
                # Calculate stop price (above upthrust high)
                upthrust_high = window.loc[window['upthrust'] == 1, 'high'].max()
                stop_price = upthrust_high * 1.02  # 2% above upthrust high
                
                # Create the pattern
                pattern = WyckoffSchematic(
                    pattern_type=WyckoffPatternType.DISTRIBUTION,
                    start_index=start_idx_pos,
                    end_index=end_idx_pos,
                    current_phase=WyckoffPhase.PHASE_E_DIST,
                    phases=phases,
                    direction="bearish",
                    strength=strength,
                    volume_confirms=volume_confirmation > 0.7,
                    target_price=target_price,
                    stop_price=stop_price
                )
                
                patterns.append(pattern)
    
    return patterns


def detect_spring(
    data: pd.DataFrame,
    lookback: int = 50,
    sensitivity: float = 0.75
) -> List[WyckoffSchematic]:
    """
    Detect Wyckoff spring patterns in price data.
    
    Args:
        data: DataFrame with OHLCV data
        lookback: Number of bars to look back
        sensitivity: Sensitivity of pattern detection (0.0-1.0)
        
    Returns:
        List of WyckoffSchematic objects representing spring patterns
    """
    patterns = []
    
    # Ensure we have enough data
    if len(data) < lookback:
        return patterns
    
    # Get the subset of data to analyze
    subset = data.iloc[-lookback:].copy()
    
    # Calculate price and volume metrics
    subset['price_range'] = subset['high'] - subset['low']
    subset['avg_volume'] = subset['volume'].rolling(window=20).mean()
    subset['volume_ratio'] = subset['volume'] / subset['avg_volume']
    
    # Calculate support levels using recent lows
    subset['support'] = subset['low'].rolling(window=10).min()
    
    # Identify potential springs
    # Spring is characterized by a brief penetration of support followed by a close above support
    subset['spring_candidate'] = (
        (subset['low'] < subset['support'].shift(1)) &
        (subset['close'] > subset['support'].shift(1)) &
        (subset['volume'] < subset['avg_volume'] * 1.2)
    ).astype(int)
    
    # Confirm springs with subsequent price action
    subset['spring_confirmed'] = 0
    for i in range(5, len(subset)):
        if subset['spring_candidate'].iloc[i-5] == 1:
            # Confirm if price moves higher after the spring
            if subset['close'].iloc[i] > subset['close'].iloc[i-5] * 1.01:
                subset.loc[subset.index[i-5], 'spring_confirmed'] = 1
    
    # Find spring patterns
    for i in range(len(subset)):
        if subset['spring_confirmed'].iloc[i] == 1:
            # We have a spring pattern
            
            # Find the start index (10 bars before spring)
            start_idx_pos = max(0, data.index.get_loc(subset.index[i]) - 10)
            
            # Find the end index (5 bars after spring)
            end_idx_pos = min(len(data) - 1, data.index.get_loc(subset.index[i]) + 5)
            
            # Calculate pattern strength
            # Strength is based on how quickly price recovers after the spring
            recovery_speed = subset['close'].iloc[i+1:i+6].mean() / subset['close'].iloc[i]
            strength = min(1.0, (recovery_speed - 1.0) * 10 * sensitivity)
            
            # Calculate target price (measured move from recent trading range)
            trading_range_high = subset['high'].iloc[i-10:i].max()
            trading_range_low = subset['low'].iloc[i-10:i].min()
            trading_range = trading_range_high - trading_range_low
            target_price = trading_range_high + trading_range
            
            # Calculate stop price (below spring low)
            spring_low = subset['low'].iloc[i]
            stop_price = spring_low * 0.98  # 2% below spring low
            
            # Create the pattern
            pattern = WyckoffSchematic(
                pattern_type=WyckoffPatternType.SPRING,
                start_index=start_idx_pos,
                end_index=end_idx_pos,
                current_phase=WyckoffPhase.PHASE_C_ACC,
                phases={WyckoffPhase.PHASE_C_ACC: (start_idx_pos, end_idx_pos)},
                direction="bullish",
                strength=strength,
                volume_confirms=subset['volume_ratio'].iloc[i] < 1.0,
                target_price=target_price,
                stop_price=stop_price
            )
            
            patterns.append(pattern)
    
    return patterns


def detect_upthrust(
    data: pd.DataFrame,
    lookback: int = 50,
    sensitivity: float = 0.75
) -> List[WyckoffSchematic]:
    """
    Detect Wyckoff upthrust patterns in price data.
    
    Args:
        data: DataFrame with OHLCV data
        lookback: Number of bars to look back
        sensitivity: Sensitivity of pattern detection (0.0-1.0)
        
    Returns:
        List of WyckoffSchematic objects representing upthrust patterns
    """
    patterns = []
    
    # Ensure we have enough data
    if len(data) < lookback:
        return patterns
    
    # Get the subset of data to analyze
    subset = data.iloc[-lookback:].copy()
    
    # Calculate price and volume metrics
    subset['price_range'] = subset['high'] - subset['low']
    subset['avg_volume'] = subset['volume'].rolling(window=20).mean()
    subset['volume_ratio'] = subset['volume'] / subset['avg_volume']
    
    # Calculate resistance levels using recent highs
    subset['resistance'] = subset['high'].rolling(window=10).max()
    
    # Identify potential upthrusts
    # Upthrust is characterized by a brief penetration of resistance followed by a close below resistance
    subset['upthrust_candidate'] = (
        (subset['high'] > subset['resistance'].shift(1)) &
        (subset['close'] < subset['resistance'].shift(1)) &
        (subset['volume'] < subset['avg_volume'] * 1.2)
    ).astype(int)
    
    # Confirm upthrusts with subsequent price action
    subset['upthrust_confirmed'] = 0
    for i in range(5, len(subset)):
        if subset['upthrust_candidate'].iloc[i-5] == 1:
            # Confirm if price moves lower after the upthrust
            if subset['close'].iloc[i] < subset['close'].iloc[i-5] * 0.99:
                subset.loc[subset.index[i-5], 'upthrust_confirmed'] = 1
    
    # Find upthrust patterns
    for i in range(len(subset)):
        if subset['upthrust_confirmed'].iloc[i] == 1:
            # We have an upthrust pattern
            
            # Find the start index (10 bars before upthrust)
            start_idx_pos = max(0, data.index.get_loc(subset.index[i]) - 10)
            
            # Find the end index (5 bars after upthrust)
            end_idx_pos = min(len(data) - 1, data.index.get_loc(subset.index[i]) + 5)
            
            # Calculate pattern strength
            # Strength is based on how quickly price falls after the upthrust
            decline_speed = subset['close'].iloc[i] / subset['close'].iloc[i+1:i+6].mean()
            strength = min(1.0, (decline_speed - 1.0) * 10 * sensitivity)
            
            # Calculate target price (measured move from recent trading range)
            trading_range_high = subset['high'].iloc[i-10:i].max()
            trading_range_low = subset['low'].iloc[i-10:i].min()
            trading_range = trading_range_high - trading_range_low
            target_price = trading_range_low - trading_range
            
            # Calculate stop price (above upthrust high)
            upthrust_high = subset['high'].iloc[i]
            stop_price = upthrust_high * 1.02  # 2% above upthrust high
            
            # Create the pattern
            pattern = WyckoffSchematic(
                pattern_type=WyckoffPatternType.UPTHRUST,
                start_index=start_idx_pos,
                end_index=end_idx_pos,
                current_phase=WyckoffPhase.PHASE_C_DIST,
                phases={WyckoffPhase.PHASE_C_DIST: (start_idx_pos, end_idx_pos)},
                direction="bearish",
                strength=strength,
                volume_confirms=subset['volume_ratio'].iloc[i] < 1.0,
                target_price=target_price,
                stop_price=stop_price
            )
            
            patterns.append(pattern)
    
    return patterns