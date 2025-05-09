"""
Volume Spread Analysis (VSA) Pattern Utilities Module.

This module provides utility functions for VSA pattern recognition.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from feature_store_service.indicators.advanced_patterns.vsa.models import (
    VSAPatternType,
    VSADirection,
    VSABar,
    VSAPattern
)


def prepare_vsa_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for VSA analysis.
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        DataFrame with VSA metrics
    """
    # Make a copy to avoid modifying the input data
    vsa_data = data.copy()
    
    # Calculate price spread
    vsa_data['spread'] = vsa_data['high'] - vsa_data['low']
    
    # Calculate close location value (0-1)
    vsa_data['close_location'] = (vsa_data['close'] - vsa_data['low']) / vsa_data['spread']
    vsa_data['close_location'] = vsa_data['close_location'].fillna(0.5)  # Handle zero spread
    
    # Calculate volume metrics
    vsa_data['volume_sma'] = vsa_data['volume'].rolling(window=20).mean()
    vsa_data['volume_ratio'] = vsa_data['volume'] / vsa_data['volume_sma']
    vsa_data['volume_delta'] = vsa_data['volume'].pct_change()
    
    # Calculate spread metrics
    vsa_data['spread_sma'] = vsa_data['spread'].rolling(window=20).mean()
    vsa_data['spread_ratio'] = vsa_data['spread'] / vsa_data['spread_sma']
    
    # Calculate trend metrics
    vsa_data['price_sma'] = vsa_data['close'].rolling(window=20).mean()
    vsa_data['trend'] = np.where(vsa_data['close'] > vsa_data['price_sma'], 1, -1)
    
    # Calculate effort vs result
    vsa_data['effort'] = vsa_data['volume_ratio']
    vsa_data['result'] = vsa_data['spread_ratio']
    vsa_data['effort_vs_result'] = vsa_data['effort'] - vsa_data['result']
    
    return vsa_data


def extract_vsa_bars(data: pd.DataFrame) -> List[VSABar]:
    """
    Extract VSA bars from DataFrame.
    
    Args:
        data: DataFrame with VSA metrics
        
    Returns:
        List of VSABar objects
    """
    bars = []
    
    for i, (idx, row) in enumerate(data.iterrows()):
        # Skip rows with NaN values
        if pd.isna(row['volume_ratio']) or pd.isna(row['spread_ratio']):
            continue
        
        bar = VSABar(
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            spread=row['spread'],
            close_location=row['close_location'],
            volume_delta=row['volume_delta'] if not pd.isna(row['volume_delta']) else 0.0,
            timestamp=idx,
            index=i
        )
        bars.append(bar)
    
    return bars


def detect_no_demand(
    bars: List[VSABar],
    lookback: int = 50,
    sensitivity: float = 0.75
) -> List[VSAPattern]:
    """
    Detect No Demand bars in VSA.
    
    No Demand bars occur in an uptrend when price attempts to move higher
    but volume decreases, indicating lack of buying interest.
    
    Args:
        bars: List of VSA bars
        lookback: Number of bars to look back
        sensitivity: Sensitivity of pattern detection (0.0-1.0)
        
    Returns:
        List of VSAPattern objects representing No Demand patterns
    """
    patterns = []
    
    # Ensure we have enough bars
    if len(bars) < lookback + 5:
        return patterns
    
    # Limit to lookback period
    recent_bars = bars[-lookback:]
    
    # Detect No Demand bars
    for i in range(5, len(recent_bars)):
        current_bar = recent_bars[i]
        prev_bars = recent_bars[i-5:i]
        
        # Check for uptrend
        is_uptrend = sum(1 for b in prev_bars if b.close > b.open) >= 3
        
        if is_uptrend:
            # Check for narrow spread and low volume
            avg_volume = sum(b.volume for b in prev_bars) / len(prev_bars)
            avg_spread = sum(b.spread for b in prev_bars) / len(prev_bars)
            
            is_narrow_spread = current_bar.spread < avg_spread * 0.8
            is_low_volume = current_bar.volume < avg_volume * 0.7
            is_up_close = current_bar.close > current_bar.open
            
            if is_narrow_spread and is_low_volume and is_up_close:
                # We have a No Demand bar
                
                # Calculate pattern strength
                # Strength is based on how much volume decreased and how narrow the spread is
                volume_decrease = 1.0 - (current_bar.volume / avg_volume)
                spread_decrease = 1.0 - (current_bar.spread / avg_spread)
                
                # Normalize strength (0.0-1.0)
                normalized_strength = min(1.0, (volume_decrease + spread_decrease) / 2 * sensitivity)
                
                # Calculate target price (measured move)
                # No Demand is bearish, so target is lower
                recent_high = max(b.high for b in prev_bars + [current_bar])
                recent_low = min(b.low for b in prev_bars + [current_bar])
                price_range = recent_high - recent_low
                target_price = current_bar.close - price_range
                
                # Calculate stop price (above recent high)
                stop_price = recent_high * 1.01  # 1% above recent high
                
                # Create the pattern
                pattern = VSAPattern(
                    pattern_type=VSAPatternType.NO_DEMAND,
                    start_index=current_bar.index - 5,
                    end_index=current_bar.index,
                    bars=prev_bars + [current_bar],
                    direction=VSADirection.BEARISH,
                    strength=normalized_strength,
                    volume_confirms=True,
                    target_price=target_price,
                    stop_price=stop_price
                )
                
                patterns.append(pattern)
    
    return patterns


def detect_no_supply(
    bars: List[VSABar],
    lookback: int = 50,
    sensitivity: float = 0.75
) -> List[VSAPattern]:
    """
    Detect No Supply bars in VSA.
    
    No Supply bars occur in a downtrend when price attempts to move lower
    but volume decreases, indicating lack of selling interest.
    
    Args:
        bars: List of VSA bars
        lookback: Number of bars to look back
        sensitivity: Sensitivity of pattern detection (0.0-1.0)
        
    Returns:
        List of VSAPattern objects representing No Supply patterns
    """
    patterns = []
    
    # Ensure we have enough bars
    if len(bars) < lookback + 5:
        return patterns
    
    # Limit to lookback period
    recent_bars = bars[-lookback:]
    
    # Detect No Supply bars
    for i in range(5, len(recent_bars)):
        current_bar = recent_bars[i]
        prev_bars = recent_bars[i-5:i]
        
        # Check for downtrend
        is_downtrend = sum(1 for b in prev_bars if b.close < b.open) >= 3
        
        if is_downtrend:
            # Check for narrow spread and low volume
            avg_volume = sum(b.volume for b in prev_bars) / len(prev_bars)
            avg_spread = sum(b.spread for b in prev_bars) / len(prev_bars)
            
            is_narrow_spread = current_bar.spread < avg_spread * 0.8
            is_low_volume = current_bar.volume < avg_volume * 0.7
            is_down_close = current_bar.close < current_bar.open
            
            if is_narrow_spread and is_low_volume and is_down_close:
                # We have a No Supply bar
                
                # Calculate pattern strength
                # Strength is based on how much volume decreased and how narrow the spread is
                volume_decrease = 1.0 - (current_bar.volume / avg_volume)
                spread_decrease = 1.0 - (current_bar.spread / avg_spread)
                
                # Normalize strength (0.0-1.0)
                normalized_strength = min(1.0, (volume_decrease + spread_decrease) / 2 * sensitivity)
                
                # Calculate target price (measured move)
                # No Supply is bullish, so target is higher
                recent_high = max(b.high for b in prev_bars + [current_bar])
                recent_low = min(b.low for b in prev_bars + [current_bar])
                price_range = recent_high - recent_low
                target_price = current_bar.close + price_range
                
                # Calculate stop price (below recent low)
                stop_price = recent_low * 0.99  # 1% below recent low
                
                # Create the pattern
                pattern = VSAPattern(
                    pattern_type=VSAPatternType.NO_SUPPLY,
                    start_index=current_bar.index - 5,
                    end_index=current_bar.index,
                    bars=prev_bars + [current_bar],
                    direction=VSADirection.BULLISH,
                    strength=normalized_strength,
                    volume_confirms=True,
                    target_price=target_price,
                    stop_price=stop_price
                )
                
                patterns.append(pattern)
    
    return patterns


def detect_stopping_volume(
    bars: List[VSABar],
    lookback: int = 50,
    sensitivity: float = 0.75
) -> List[VSAPattern]:
    """
    Detect Stopping Volume bars in VSA.
    
    Stopping Volume occurs at the end of a downtrend when a wide-range down bar
    closes well off the lows with high volume, indicating selling exhaustion.
    
    Args:
        bars: List of VSA bars
        lookback: Number of bars to look back
        sensitivity: Sensitivity of pattern detection (0.0-1.0)
        
    Returns:
        List of VSAPattern objects representing Stopping Volume patterns
    """
    patterns = []
    
    # Ensure we have enough bars
    if len(bars) < lookback + 5:
        return patterns
    
    # Limit to lookback period
    recent_bars = bars[-lookback:]
    
    # Detect Stopping Volume bars
    for i in range(5, len(recent_bars)):
        current_bar = recent_bars[i]
        prev_bars = recent_bars[i-5:i]
        
        # Check for downtrend
        is_downtrend = sum(1 for b in prev_bars if b.close < b.open) >= 3
        
        if is_downtrend:
            # Check for wide spread, high volume, and close off the lows
            avg_volume = sum(b.volume for b in prev_bars) / len(prev_bars)
            avg_spread = sum(b.spread for b in prev_bars) / len(prev_bars)
            
            is_wide_spread = current_bar.spread > avg_spread * 1.2
            is_high_volume = current_bar.volume > avg_volume * 1.5
            is_close_off_lows = current_bar.close_location > 0.4  # Close in upper 60% of bar
            
            if is_wide_spread and is_high_volume and is_close_off_lows:
                # We have a Stopping Volume bar
                
                # Calculate pattern strength
                # Strength is based on volume increase and close location
                volume_increase = current_bar.volume / avg_volume
                close_strength = current_bar.close_location
                
                # Normalize strength (0.0-1.0)
                normalized_strength = min(1.0, (volume_increase / 3 + close_strength) / 2 * sensitivity)
                
                # Calculate target price (measured move)
                # Stopping Volume is bullish, so target is higher
                recent_high = max(b.high for b in prev_bars)
                recent_low = current_bar.low  # Use the stopping volume low
                price_range = recent_high - recent_low
                target_price = current_bar.close + price_range
                
                # Calculate stop price (below stopping volume low)
                stop_price = current_bar.low * 0.99  # 1% below stopping volume low
                
                # Create the pattern
                pattern = VSAPattern(
                    pattern_type=VSAPatternType.STOPPING_VOLUME,
                    start_index=current_bar.index - 5,
                    end_index=current_bar.index,
                    bars=prev_bars + [current_bar],
                    direction=VSADirection.BULLISH,
                    strength=normalized_strength,
                    volume_confirms=True,
                    target_price=target_price,
                    stop_price=stop_price
                )
                
                patterns.append(pattern)
    
    return patterns


def detect_climactic_volume(
    bars: List[VSABar],
    lookback: int = 50,
    sensitivity: float = 0.75
) -> List[VSAPattern]:
    """
    Detect Climactic Volume bars in VSA.
    
    Climactic Volume occurs at the end of a strong move with extremely high volume
    and wide range, indicating potential exhaustion of the current trend.
    
    Args:
        bars: List of VSA bars
        lookback: Number of bars to look back
        sensitivity: Sensitivity of pattern detection (0.0-1.0)
        
    Returns:
        List of VSAPattern objects representing Climactic Volume patterns
    """
    patterns = []
    
    # Ensure we have enough bars
    if len(bars) < lookback + 10:
        return patterns
    
    # Limit to lookback period
    recent_bars = bars[-lookback:]
    
    # Detect Climactic Volume bars
    for i in range(10, len(recent_bars)):
        current_bar = recent_bars[i]
        prev_bars = recent_bars[i-10:i]
        
        # Check for a strong trend
        up_bars = sum(1 for b in prev_bars if b.close > b.open)
        down_bars = len(prev_bars) - up_bars
        is_uptrend = up_bars >= 7
        is_downtrend = down_bars >= 7
        
        if is_uptrend or is_downtrend:
            # Check for extremely high volume and wide range
            avg_volume = sum(b.volume for b in prev_bars) / len(prev_bars)
            max_volume = max(b.volume for b in prev_bars)
            avg_spread = sum(b.spread for b in prev_bars) / len(prev_bars)
            
            is_extreme_volume = current_bar.volume > max_volume and current_bar.volume > avg_volume * 2
            is_wide_spread = current_bar.spread > avg_spread * 1.5
            
            if is_extreme_volume and is_wide_spread:
                # We have a Climactic Volume bar
                
                # Determine direction based on trend and close location
                if is_uptrend and current_bar.close_location < 0.3:
                    direction = VSADirection.BEARISH  # Buying climax
                elif is_downtrend and current_bar.close_location > 0.7:
                    direction = VSADirection.BULLISH  # Selling climax
                else:
                    direction = VSADirection.NEUTRAL  # Indeterminate
                
                # Calculate pattern strength
                # Strength is based on volume extremity and spread width
                volume_extremity = current_bar.volume / avg_volume
                spread_extremity = current_bar.spread / avg_spread
                
                # Normalize strength (0.0-1.0)
                normalized_strength = min(1.0, (volume_extremity / 4 + spread_extremity / 2) / 2 * sensitivity)
                
                # Calculate target price (measured move)
                if direction == VSADirection.BEARISH:
                    # Buying climax is bearish, so target is lower
                    recent_low = min(b.low for b in prev_bars)
                    price_range = current_bar.high - recent_low
                    target_price = current_bar.close - price_range
                    stop_price = current_bar.high * 1.01  # 1% above climax high
                elif direction == VSADirection.BULLISH:
                    # Selling climax is bullish, so target is higher
                    recent_high = max(b.high for b in prev_bars)
                    price_range = recent_high - current_bar.low
                    target_price = current_bar.close + price_range
                    stop_price = current_bar.low * 0.99  # 1% below climax low
                else:
                    # Neutral, no clear target
                    target_price = None
                    stop_price = None
                
                # Create the pattern
                pattern = VSAPattern(
                    pattern_type=VSAPatternType.CLIMACTIC_VOLUME,
                    start_index=current_bar.index - 10,
                    end_index=current_bar.index,
                    bars=prev_bars + [current_bar],
                    direction=direction,
                    strength=normalized_strength,
                    volume_confirms=True,
                    target_price=target_price,
                    stop_price=stop_price
                )
                
                patterns.append(pattern)
    
    return patterns


def detect_effort_vs_result(
    bars: List[VSABar],
    lookback: int = 50,
    sensitivity: float = 0.75
) -> List[VSAPattern]:
    """
    Detect Effort vs Result imbalances in VSA.
    
    Effort vs Result imbalance occurs when volume (effort) doesn't match
    the price movement (result), indicating potential trend change.
    
    Args:
        bars: List of VSA bars
        lookback: Number of bars to look back
        sensitivity: Sensitivity of pattern detection (0.0-1.0)
        
    Returns:
        List of VSAPattern objects representing Effort vs Result patterns
    """
    patterns = []
    
    # Ensure we have enough bars
    if len(bars) < lookback + 5:
        return patterns
    
    # Limit to lookback period
    recent_bars = bars[-lookback:]
    
    # Detect Effort vs Result imbalances
    for i in range(5, len(recent_bars)):
        current_bar = recent_bars[i]
        prev_bars = recent_bars[i-5:i]
        
        # Calculate average volume and spread
        avg_volume = sum(b.volume for b in prev_bars) / len(prev_bars)
        avg_spread = sum(b.spread for b in prev_bars) / len(prev_bars)
        
        # Check for high effort (volume) with poor result (spread/close)
        is_high_volume = current_bar.volume > avg_volume * 1.5
        
        # Case 1: High volume up bar with poor close (bearish)
        if is_high_volume and current_bar.close > current_bar.open and current_bar.close_location < 0.5:
            # High effort with poor result (bearish)
            direction = VSADirection.BEARISH
            
            # Calculate pattern strength
            volume_effort = current_bar.volume / avg_volume
            result_weakness = 1.0 - current_bar.close_location
            
            # Normalize strength (0.0-1.0)
            normalized_strength = min(1.0, (volume_effort / 3 + result_weakness) / 2 * sensitivity)
            
            # Calculate target price (measured move)
            recent_low = min(b.low for b in prev_bars)
            price_range = current_bar.high - recent_low
            target_price = current_bar.close - price_range
            
            # Calculate stop price (above bar high)
            stop_price = current_bar.high * 1.01  # 1% above bar high
            
            # Create the pattern
            pattern = VSAPattern(
                pattern_type=VSAPatternType.EFFORT_VS_RESULT,
                start_index=current_bar.index - 5,
                end_index=current_bar.index,
                bars=prev_bars + [current_bar],
                direction=direction,
                strength=normalized_strength,
                volume_confirms=True,
                target_price=target_price,
                stop_price=stop_price
            )
            
            patterns.append(pattern)
        
        # Case 2: High volume down bar with poor close (bullish)
        elif is_high_volume and current_bar.close < current_bar.open and current_bar.close_location > 0.5:
            # High effort with poor result (bullish)
            direction = VSADirection.BULLISH
            
            # Calculate pattern strength
            volume_effort = current_bar.volume / avg_volume
            result_weakness = current_bar.close_location
            
            # Normalize strength (0.0-1.0)
            normalized_strength = min(1.0, (volume_effort / 3 + result_weakness) / 2 * sensitivity)
            
            # Calculate target price (measured move)
            recent_high = max(b.high for b in prev_bars)
            price_range = recent_high - current_bar.low
            target_price = current_bar.close + price_range
            
            # Calculate stop price (below bar low)
            stop_price = current_bar.low * 0.99  # 1% below bar low
            
            # Create the pattern
            pattern = VSAPattern(
                pattern_type=VSAPatternType.EFFORT_VS_RESULT,
                start_index=current_bar.index - 5,
                end_index=current_bar.index,
                bars=prev_bars + [current_bar],
                direction=direction,
                strength=normalized_strength,
                volume_confirms=True,
                target_price=target_price,
                stop_price=stop_price
            )
            
            patterns.append(pattern)
    
    return patterns


def detect_trap_move(
    bars: List[VSABar],
    lookback: int = 50,
    sensitivity: float = 0.75
) -> List[VSAPattern]:
    """
    Detect Trap Moves in VSA.
    
    Trap Moves occur when price breaks a significant level but quickly reverses,
    trapping traders who entered on the breakout.
    
    Args:
        bars: List of VSA bars
        lookback: Number of bars to look back
        sensitivity: Sensitivity of pattern detection (0.0-1.0)
        
    Returns:
        List of VSAPattern objects representing Trap Move patterns
    """
    patterns = []
    
    # Ensure we have enough bars
    if len(bars) < lookback + 10:
        return patterns
    
    # Limit to lookback period
    recent_bars = bars[-lookback:]
    
    # Find significant levels (recent highs/lows)
    for i in range(10, len(recent_bars) - 2):
        # Look for a potential breakout bar
        current_bar = recent_bars[i]
        prev_bars = recent_bars[i-10:i]
        next_bars = recent_bars[i+1:i+3]  # 2 bars after breakout
        
        # Find recent high and low
        recent_high = max(b.high for b in prev_bars)
        recent_low = min(b.low for b in prev_bars)
        
        # Check for breakout above recent high
        if current_bar.high > recent_high and current_bar.close > recent_high:
            # Check for reversal in the next 2 bars
            if any(b.close < recent_high for b in next_bars):
                # We have a bearish trap (false breakout)
                
                # Calculate pattern strength
                breakout_size = (current_bar.high - recent_high) / recent_high
                reversal_size = (current_bar.close - next_bars[-1].close) / current_bar.close
                
                # Normalize strength (0.0-1.0)
                normalized_strength = min(1.0, (breakout_size + reversal_size) * 10 * sensitivity)
                
                # Calculate target price (measured move)
                price_range = recent_high - recent_low
                target_price = recent_high - price_range
                
                # Calculate stop price (above trap high)
                stop_price = current_bar.high * 1.01  # 1% above trap high
                
                # Create the pattern
                pattern = VSAPattern(
                    pattern_type=VSAPatternType.TRAP_MOVE,
                    start_index=current_bar.index - 10,
                    end_index=current_bar.index + 2,
                    bars=prev_bars + [current_bar] + next_bars,
                    direction=VSADirection.BEARISH,
                    strength=normalized_strength,
                    volume_confirms=current_bar.volume > sum(b.volume for b in prev_bars) / len(prev_bars),
                    target_price=target_price,
                    stop_price=stop_price
                )
                
                patterns.append(pattern)
        
        # Check for breakout below recent low
        elif current_bar.low < recent_low and current_bar.close < recent_low:
            # Check for reversal in the next 2 bars
            if any(b.close > recent_low for b in next_bars):
                # We have a bullish trap (false breakdown)
                
                # Calculate pattern strength
                breakout_size = (recent_low - current_bar.low) / recent_low
                reversal_size = (next_bars[-1].close - current_bar.close) / current_bar.close
                
                # Normalize strength (0.0-1.0)
                normalized_strength = min(1.0, (breakout_size + reversal_size) * 10 * sensitivity)
                
                # Calculate target price (measured move)
                price_range = recent_high - recent_low
                target_price = recent_low + price_range
                
                # Calculate stop price (below trap low)
                stop_price = current_bar.low * 0.99  # 1% below trap low
                
                # Create the pattern
                pattern = VSAPattern(
                    pattern_type=VSAPatternType.TRAP_MOVE,
                    start_index=current_bar.index - 10,
                    end_index=current_bar.index + 2,
                    bars=prev_bars + [current_bar] + next_bars,
                    direction=VSADirection.BULLISH,
                    strength=normalized_strength,
                    volume_confirms=current_bar.volume > sum(b.volume for b in prev_bars) / len(prev_bars),
                    target_price=target_price,
                    stop_price=stop_price
                )
                
                patterns.append(pattern)
    
    return patterns