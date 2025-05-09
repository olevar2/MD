"""
Elliott Wave Validators Module.

This module provides functions for validating Elliott Wave patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from analysis_engine.analysis.advanced_ta.base import ConfidenceLevel
from analysis_engine.analysis.advanced_ta.elliott_wave.utils import calculate_wave_sharpness


def validate_elliott_rules(
    df: pd.DataFrame, 
    candidate: Dict[str, Tuple[int, float, int, float]],
    parameters: Dict[str, Any]
) -> Tuple[bool, float]:
    """
    Validate a candidate wave count against Elliott Wave rules
    
    Args:
        df: DataFrame with OHLCV data
        candidate: Candidate wave count
        parameters: Dictionary of parameters for validation
        
    Returns:
        Tuple of (is_valid, confidence)
    """
    # Initialize confidence score
    confidence_score = 1.0
    price_col = parameters.get("price_column", "close")
    
    # Wave 1: Extracting data
    wave1_start_idx, wave1_start_price, wave1_end_idx, wave1_end_price = candidate["ONE"]
    wave2_start_idx, wave2_start_price, wave2_end_idx, wave2_end_price = candidate["TWO"]
    wave3_start_idx, wave3_start_price, wave3_end_idx, wave3_end_price = candidate["THREE"]
    wave4_start_idx, wave4_start_price, wave4_end_idx, wave4_end_price = candidate["FOUR"]
    wave5_start_idx, wave5_start_price, wave5_end_idx, wave5_end_price = candidate["FIVE"]
    
    # Calculate wave lengths (in price units)
    wave1_length = abs(wave1_end_price - wave1_start_price)
    wave2_length = abs(wave2_end_price - wave2_start_price)
    wave3_length = abs(wave3_end_price - wave3_start_price)
    wave4_length = abs(wave4_end_price - wave4_start_price)
    wave5_length = abs(wave5_end_price - wave5_start_price)
    
    # Calculate wave durations (in bars)
    wave1_duration = wave1_end_idx - wave1_start_idx
    wave2_duration = wave2_end_idx - wave2_start_idx
    wave3_duration = wave3_end_idx - wave3_start_idx
    wave4_duration = wave4_end_idx - wave4_start_idx
    wave5_duration = wave5_end_idx - wave5_start_idx
    
    # Check if uptrend or downtrend
    is_uptrend = wave1_end_price > wave1_start_price
    
    # Rule 1: Wave 2 should not retrace more than 100% of wave 1
    wave2_retracement = wave2_length / wave1_length
    if wave2_retracement > 1.0:
        return False, 0.0
    
    # Rule 2: Wave 3 should never be the shortest of waves 1, 3 and 5
    if wave3_length < wave1_length and wave3_length < wave5_length:
        return False, 0.0
        
    # Rule 3: Wave 4 should not overlap wave 1, except in diagonal patterns
    if is_uptrend:
        if wave4_end_price < wave1_end_price and not check_if_diagonal(candidate):
            return False, 0.0
    else:
        if wave4_end_price > wave1_end_price and not check_if_diagonal(candidate):
            return False, 0.0
            
    # Rule 4: Minimum length requirements for each wave
    min_wave_length = parameters.get("min_wave_length", 5)
    if (wave1_duration < min_wave_length or
        wave2_duration < min_wave_length or
        wave3_duration < min_wave_length or
        wave4_duration < min_wave_length or
        wave5_duration < min_wave_length):
        confidence_score *= 0.7  # Reduce confidence if waves are short
        
    # Guideline 1: Wave 3 is often extended (longest and strongest)
    if not (wave3_length > wave1_length and wave3_length > wave5_length):
        confidence_score *= 0.9
        
    # Guideline 2: Wave relationships - common Fibonacci ratios
    # Wave 3 is often 1.618 * wave 1
    ideal_wave3 = wave1_length * 1.618
    wave3_ratio_diff = abs(wave3_length - ideal_wave3) / ideal_wave3
    if wave3_ratio_diff > 0.2:  # Allow 20% tolerance
        confidence_score *= 0.9
        
    # Wave 5 is often 0.618 * wave 1
    ideal_wave5 = wave1_length * 0.618
    wave5_ratio_diff = abs(wave5_length - ideal_wave5) / ideal_wave5
    if wave5_ratio_diff > 0.3:  # Allow 30% tolerance
        confidence_score *= 0.9
        
    # Guideline 3: Alternation between waves 2 and 4
    if parameters.get("alternation_check", True):
        # Check if wave 2 and wave 4 are different in structure (sharp vs flat)
        wave2_sharpness = calculate_wave_sharpness(df, wave2_start_idx, wave2_end_idx, price_col)
        wave4_sharpness = calculate_wave_sharpness(df, wave4_start_idx, wave4_end_idx, price_col)
        
        if abs(wave2_sharpness - wave4_sharpness) < 0.2:  # Not much alternation
            confidence_score *= 0.8
    
    # Adjust final confidence based on pattern complexity
    if check_if_extended(candidate):
        confidence_score *= 1.1  # Bonus for detecting extended waves
        
    if check_if_diagonal(candidate):
        confidence_score *= 0.9  # Slightly reduce confidence for diagonal patterns
        
    # Cap confidence at 1.0
    confidence_score = min(confidence_score, 1.0)
        
    return True, confidence_score


def check_if_extended(candidate: Dict[str, Tuple[int, float, int, float]]) -> bool:
    """
    Check if any wave is extended (significantly larger than the others)
    
    Args:
        candidate: Candidate wave count
        
    Returns:
        True if an extended wave is detected, False otherwise
    """
    wave1_start_idx, wave1_start_price, wave1_end_idx, wave1_end_price = candidate["ONE"]
    wave3_start_idx, wave3_start_price, wave3_end_idx, wave3_end_price = candidate["THREE"]
    wave5_start_idx, wave5_start_price, wave5_end_idx, wave5_end_price = candidate["FIVE"]
    
    wave1_length = abs(wave1_end_price - wave1_start_price)
    wave3_length = abs(wave3_end_price - wave3_start_price)
    wave5_length = abs(wave5_end_price - wave5_start_price)
    
    # Extended wave is typically at least 1.618 times the next largest wave
    if wave3_length > (1.618 * max(wave1_length, wave5_length)):
        return True
        
    if wave5_length > (1.618 * max(wave1_length, wave3_length)):
        return True
        
    if wave1_length > (1.618 * max(wave3_length, wave5_length)):
        return True
    
    return False


def check_if_diagonal(candidate: Dict[str, Tuple[int, float, int, float]]) -> bool:
    """
    Check if the pattern forms a diagonal (wedge-shaped) pattern
    
    Args:
        candidate: Candidate wave count
        
    Returns:
        True if a diagonal pattern is detected, False otherwise
    """
    # Extract wave end points
    wave1_end_idx, wave1_end_price = candidate["ONE"][2:4]
    wave2_end_idx, wave2_end_price = candidate["TWO"][2:4]
    wave3_end_idx, wave3_end_price = candidate["THREE"][2:4]
    wave4_end_idx, wave4_end_price = candidate["FOUR"][2:4]
    wave5_end_idx, wave5_end_price = candidate["FIVE"][2:4]
    
    # Calculate slopes of trendlines connecting wave 1-3-5 and 2-4
    try:
        # Upper trendline: connects waves 1, 3, 5 ends
        upper_slope1 = (wave3_end_price - wave1_end_price) / (wave3_end_idx - wave1_end_idx)
        upper_slope2 = (wave5_end_price - wave3_end_price) / (wave5_end_idx - wave3_end_idx)
        
        # Lower trendline: connects waves 2, 4 ends
        lower_slope = (wave4_end_price - wave2_end_price) / (wave4_end_idx - wave2_end_idx)
        
        # In a diagonal, the slopes should be in the same direction (converging)
        if (upper_slope1 > 0 and upper_slope2 > 0 and lower_slope > 0) or \
           (upper_slope1 < 0 and upper_slope2 < 0 and lower_slope < 0):
            
            # Check for convergence: slopes should be getting less steep
            if abs(upper_slope2) < abs(upper_slope1):
                return True
    except:
        pass
        
    return False


def map_confidence_to_level(confidence: float) -> ConfidenceLevel:
    """
    Map a confidence score to a ConfidenceLevel enum
    
    Args:
        confidence: Confidence score (0.0 to 1.0)
        
    Returns:
        ConfidenceLevel enum value
    """
    if confidence >= 0.8:
        return ConfidenceLevel.HIGH
    elif confidence >= 0.6:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW