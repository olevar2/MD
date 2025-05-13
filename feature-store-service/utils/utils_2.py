"""
Renko Pattern Utilities Module.

This module provides utility functions for Renko pattern analysis.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np

from models.models_3 import (
    RenkoBrick,
    RenkoDirection,
    RenkoPatternType,
    RenkoPattern
)


def detect_renko_reversal(
    bricks: List[RenkoBrick],
    min_trend_length: int = 3,
    lookback: int = 10
) -> List[RenkoPattern]:
    """
    Detect reversal patterns in Renko bricks.
    
    Args:
        bricks: List of Renko bricks
        min_trend_length: Minimum number of bricks in the same direction before a reversal
        lookback: Maximum number of bricks to look back
        
    Returns:
        List of detected reversal patterns
    """
    if len(bricks) < min_trend_length + 1:
        return []
    
    patterns = []
    
    # Look for reversals
    for i in range(min_trend_length, len(bricks)):
        # Limit lookback
        start_idx = max(0, i - lookback)
        
        # Check for reversal
        if bricks[i].direction != bricks[i-1].direction:
            # Count consecutive bricks in the previous direction
            count = 0
            for j in range(i-1, start_idx-1, -1):
                if bricks[j].direction == bricks[i-1].direction:
                    count += 1
                else:
                    break
            
            if count >= min_trend_length:
                # Found a reversal pattern
                pattern_bricks = bricks[i-count:i+1]
                
                # Determine pattern direction
                direction = "bullish" if bricks[i].direction == RenkoDirection.UP else "bearish"
                
                # Calculate pattern strength (0.0-1.0)
                strength = min(1.0, count / 10)  # Normalize to 0.0-1.0
                
                # Calculate target and stop prices
                if direction == "bullish":
                    # Bullish reversal
                    pattern_height = sum(brick.size for brick in pattern_bricks if brick.direction == RenkoDirection.UP)
                    target_price = bricks[i].close_price + pattern_height
                    stop_price = bricks[i-1].close_price - pattern_bricks[0].size
                else:
                    # Bearish reversal
                    pattern_height = sum(brick.size for brick in pattern_bricks if brick.direction == RenkoDirection.DOWN)
                    target_price = bricks[i].close_price - pattern_height
                    stop_price = bricks[i-1].close_price + pattern_bricks[0].size
                
                # Create pattern object
                pattern = RenkoPattern(
                    pattern_type=RenkoPatternType.REVERSAL,
                    start_index=i-count,
                    end_index=i,
                    bricks=pattern_bricks,
                    direction=direction,
                    strength=strength,
                    target_price=target_price,
                    stop_price=stop_price
                )
                
                patterns.append(pattern)
    
    return patterns


def detect_renko_breakout(
    bricks: List[RenkoBrick],
    consolidation_max_range: int = 3,
    min_consolidation_length: int = 4,
    lookback: int = 20
) -> List[RenkoPattern]:
    """
    Detect breakout patterns in Renko bricks.
    
    Args:
        bricks: List of Renko bricks
        consolidation_max_range: Maximum range of consolidation in brick sizes
        min_consolidation_length: Minimum number of bricks in consolidation
        lookback: Maximum number of bricks to look back
        
    Returns:
        List of detected breakout patterns
    """
    if len(bricks) < min_consolidation_length + 1:
        return []
    
    patterns = []
    
    # Look for breakouts
    for i in range(min_consolidation_length, len(bricks)):
        # Limit lookback
        start_idx = max(0, i - lookback)
        
        # Check for potential consolidation followed by breakout
        if i >= 2 and bricks[i].direction == bricks[i-1].direction:
            # Find the consolidation range
            consolidation_high = float('-inf')
            consolidation_low = float('inf')
            consolidation_start = i - 1
            
            # Scan backward to find consolidation
            for j in range(i-2, start_idx-1, -1):
                current_price = bricks[j].close_price
                consolidation_high = max(consolidation_high, current_price)
                consolidation_low = min(consolidation_low, current_price)
                
                # Check if we're still in consolidation range
                if consolidation_high - consolidation_low <= consolidation_max_range * bricks[j].size:
                    consolidation_start = j
                else:
                    break
            
            consolidation_length = i - 1 - consolidation_start
            
            if consolidation_length >= min_consolidation_length:
                # Found a consolidation followed by breakout
                pattern_bricks = bricks[consolidation_start:i+1]
                
                # Determine pattern direction
                direction = "bullish" if bricks[i].direction == RenkoDirection.UP else "bearish"
                
                # Calculate pattern strength (0.0-1.0)
                strength = min(1.0, consolidation_length / 15)  # Normalize to 0.0-1.0
                
                # Calculate target and stop prices
                consolidation_height = consolidation_high - consolidation_low
                
                if direction == "bullish":
                    # Bullish breakout
                    target_price = bricks[i].close_price + consolidation_height
                    stop_price = consolidation_low
                else:
                    # Bearish breakout
                    target_price = bricks[i].close_price - consolidation_height
                    stop_price = consolidation_high
                
                # Create pattern object
                pattern = RenkoPattern(
                    pattern_type=RenkoPatternType.BREAKOUT,
                    start_index=consolidation_start,
                    end_index=i,
                    bricks=pattern_bricks,
                    direction=direction,
                    strength=strength,
                    target_price=target_price,
                    stop_price=stop_price
                )
                
                patterns.append(pattern)
    
    return patterns


def detect_renko_double_formation(
    bricks: List[RenkoBrick],
    max_deviation: float = 0.1,
    min_pattern_size: int = 5,
    lookback: int = 30
) -> List[RenkoPattern]:
    """
    Detect double top/bottom patterns in Renko bricks.
    
    Args:
        bricks: List of Renko bricks
        max_deviation: Maximum deviation between the two tops/bottoms (as percentage of brick size)
        min_pattern_size: Minimum number of bricks in the pattern
        lookback: Maximum number of bricks to look back
        
    Returns:
        List of detected double top/bottom patterns
    """
    if len(bricks) < min_pattern_size:
        return []
    
    patterns = []
    
    # Look for double tops/bottoms
    for i in range(min_pattern_size, len(bricks)):
        # Limit lookback
        start_idx = max(0, i - lookback)
        
        # Check for potential double top/bottom
        if i >= 2 and bricks[i].direction != bricks[i-1].direction:
            # Find potential first top/bottom
            first_extreme_idx = None
            first_extreme_price = None
            
            # Scan backward to find first extreme
            for j in range(i-2, start_idx-1, -1):
                if j > 0 and bricks[j].direction != bricks[j-1].direction:
                    if bricks[j-1].direction != bricks[i-1].direction:
                        # Found a potential first extreme
                        first_extreme_idx = j
                        first_extreme_price = bricks[j].close_price
                        break
            
            if first_extreme_idx is not None:
                # Check if the two extremes are at similar price levels
                second_extreme_price = bricks[i-1].close_price
                avg_brick_size = sum(brick.size for brick in bricks[first_extreme_idx:i]) / (i - first_extreme_idx)
                
                if abs(second_extreme_price - first_extreme_price) <= max_deviation * avg_brick_size:
                    # Found a double top/bottom pattern
                    pattern_bricks = bricks[first_extreme_idx:i+1]
                    
                    # Determine pattern type and direction
                    if bricks[i].direction == RenkoDirection.UP:
                        # Double bottom (bullish)
                        pattern_type = RenkoPatternType.DOUBLE_BOTTOM
                        direction = "bullish"
                    else:
                        # Double top (bearish)
                        pattern_type = RenkoPatternType.DOUBLE_TOP
                        direction = "bearish"
                    
                    # Calculate pattern strength (0.0-1.0)
                    pattern_length = i - first_extreme_idx
                    strength = min(1.0, pattern_length / 20)  # Normalize to 0.0-1.0
                    
                    # Calculate target and stop prices
                    pattern_height = max(brick.close_price for brick in pattern_bricks) - min(brick.close_price for brick in pattern_bricks)
                    
                    if direction == "bullish":
                        # Double bottom
                        target_price = bricks[i].close_price + pattern_height
                        stop_price = min(brick.close_price for brick in pattern_bricks) - avg_brick_size
                    else:
                        # Double top
                        target_price = bricks[i].close_price - pattern_height
                        stop_price = max(brick.close_price for brick in pattern_bricks) + avg_brick_size
                    
                    # Create pattern object
                    pattern = RenkoPattern(
                        pattern_type=pattern_type,
                        start_index=first_extreme_idx,
                        end_index=i,
                        bricks=pattern_bricks,
                        direction=direction,
                        strength=strength,
                        target_price=target_price,
                        stop_price=stop_price
                    )
                    
                    patterns.append(pattern)
    
    return patterns