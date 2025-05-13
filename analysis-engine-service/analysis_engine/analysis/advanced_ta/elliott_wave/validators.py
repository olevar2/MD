"""
Elliott Wave Validators Module.

This module provides functions for validating Elliott Wave patterns.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from analysis_engine.analysis.advanced_ta.base import ConfidenceLevel
from analysis_engine.analysis.advanced_ta.elliott_wave.utils import calculate_wave_sharpness


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

def validate_elliott_rules(df: pd.DataFrame, candidate: Dict[str, Tuple[int,
    float, int, float]], parameters: Dict[str, Any]) ->Tuple[bool, float]:
    """
    Validate a candidate wave count against Elliott Wave rules
    
    Args:
        df: DataFrame with OHLCV data
        candidate: Candidate wave count
        parameters: Dictionary of parameters for validation
        
    Returns:
        Tuple of (is_valid, confidence)
    """
    confidence_score = 1.0
    price_col = parameters.get('price_column', 'close')
    wave1_start_idx, wave1_start_price, wave1_end_idx, wave1_end_price = (
        candidate['ONE'])
    wave2_start_idx, wave2_start_price, wave2_end_idx, wave2_end_price = (
        candidate['TWO'])
    wave3_start_idx, wave3_start_price, wave3_end_idx, wave3_end_price = (
        candidate['THREE'])
    wave4_start_idx, wave4_start_price, wave4_end_idx, wave4_end_price = (
        candidate['FOUR'])
    wave5_start_idx, wave5_start_price, wave5_end_idx, wave5_end_price = (
        candidate['FIVE'])
    wave1_length = abs(wave1_end_price - wave1_start_price)
    wave2_length = abs(wave2_end_price - wave2_start_price)
    wave3_length = abs(wave3_end_price - wave3_start_price)
    wave4_length = abs(wave4_end_price - wave4_start_price)
    wave5_length = abs(wave5_end_price - wave5_start_price)
    wave1_duration = wave1_end_idx - wave1_start_idx
    wave2_duration = wave2_end_idx - wave2_start_idx
    wave3_duration = wave3_end_idx - wave3_start_idx
    wave4_duration = wave4_end_idx - wave4_start_idx
    wave5_duration = wave5_end_idx - wave5_start_idx
    is_uptrend = wave1_end_price > wave1_start_price
    wave2_retracement = wave2_length / wave1_length
    if wave2_retracement > 1.0:
        return False, 0.0
    if wave3_length < wave1_length and wave3_length < wave5_length:
        return False, 0.0
    if is_uptrend:
        if wave4_end_price < wave1_end_price and not check_if_diagonal(
            candidate):
            return False, 0.0
    elif wave4_end_price > wave1_end_price and not check_if_diagonal(candidate
        ):
        return False, 0.0
    min_wave_length = parameters.get('min_wave_length', 5)
    if (wave1_duration < min_wave_length or wave2_duration <
        min_wave_length or wave3_duration < min_wave_length or 
        wave4_duration < min_wave_length or wave5_duration < min_wave_length):
        confidence_score *= 0.7
    if not (wave3_length > wave1_length and wave3_length > wave5_length):
        confidence_score *= 0.9
    ideal_wave3 = wave1_length * 1.618
    wave3_ratio_diff = abs(wave3_length - ideal_wave3) / ideal_wave3
    if wave3_ratio_diff > 0.2:
        confidence_score *= 0.9
    ideal_wave5 = wave1_length * 0.618
    wave5_ratio_diff = abs(wave5_length - ideal_wave5) / ideal_wave5
    if wave5_ratio_diff > 0.3:
        confidence_score *= 0.9
    if parameters.get('alternation_check', True):
        wave2_sharpness = calculate_wave_sharpness(df, wave2_start_idx,
            wave2_end_idx, price_col)
        wave4_sharpness = calculate_wave_sharpness(df, wave4_start_idx,
            wave4_end_idx, price_col)
        if abs(wave2_sharpness - wave4_sharpness) < 0.2:
            confidence_score *= 0.8
    if check_if_extended(candidate):
        confidence_score *= 1.1
    if check_if_diagonal(candidate):
        confidence_score *= 0.9
    confidence_score = min(confidence_score, 1.0)
    return True, confidence_score


def check_if_extended(candidate: Dict[str, Tuple[int, float, int, float]]
    ) ->bool:
    """
    Check if any wave is extended (significantly larger than the others)
    
    Args:
        candidate: Candidate wave count
        
    Returns:
        True if an extended wave is detected, False otherwise
    """
    wave1_start_idx, wave1_start_price, wave1_end_idx, wave1_end_price = (
        candidate['ONE'])
    wave3_start_idx, wave3_start_price, wave3_end_idx, wave3_end_price = (
        candidate['THREE'])
    wave5_start_idx, wave5_start_price, wave5_end_idx, wave5_end_price = (
        candidate['FIVE'])
    wave1_length = abs(wave1_end_price - wave1_start_price)
    wave3_length = abs(wave3_end_price - wave3_start_price)
    wave5_length = abs(wave5_end_price - wave5_start_price)
    if wave3_length > 1.618 * max(wave1_length, wave5_length):
        return True
    if wave5_length > 1.618 * max(wave1_length, wave3_length):
        return True
    if wave1_length > 1.618 * max(wave3_length, wave5_length):
        return True
    return False


@with_exception_handling
def check_if_diagonal(candidate: Dict[str, Tuple[int, float, int, float]]
    ) ->bool:
    """
    Check if the pattern forms a diagonal (wedge-shaped) pattern
    
    Args:
        candidate: Candidate wave count
        
    Returns:
        True if a diagonal pattern is detected, False otherwise
    """
    wave1_end_idx, wave1_end_price = candidate['ONE'][2:4]
    wave2_end_idx, wave2_end_price = candidate['TWO'][2:4]
    wave3_end_idx, wave3_end_price = candidate['THREE'][2:4]
    wave4_end_idx, wave4_end_price = candidate['FOUR'][2:4]
    wave5_end_idx, wave5_end_price = candidate['FIVE'][2:4]
    try:
        upper_slope1 = (wave3_end_price - wave1_end_price) / (wave3_end_idx -
            wave1_end_idx)
        upper_slope2 = (wave5_end_price - wave3_end_price) / (wave5_end_idx -
            wave3_end_idx)
        lower_slope = (wave4_end_price - wave2_end_price) / (wave4_end_idx -
            wave2_end_idx)
        if (upper_slope1 > 0 and upper_slope2 > 0 and lower_slope > 0 or 
            upper_slope1 < 0 and upper_slope2 < 0 and lower_slope < 0):
            if abs(upper_slope2) < abs(upper_slope1):
                return True
    except:
        pass
    return False


def map_confidence_to_level(confidence: float) ->ConfidenceLevel:
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
