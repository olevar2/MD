"""
Fibonacci utilities module.

This module provides utility functions for Fibonacci analysis.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import math


def generate_fibonacci_sequence(n: int, extended: bool = False) -> List[int]:
    """
    Generate Fibonacci sequence.
    
    Args:
        n: Number of Fibonacci numbers to generate
        extended: Whether to include 0 and start with [0, 1, 1, 2, 3, 5, ...]
            instead of [1, 2, 3, 5, 8, ...]
            
    Returns:
        List of Fibonacci numbers
    """
    if extended:
        # Extended sequence starting with 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...
        sequence = [0, 1, 1]
        while len(sequence) < n + 2:  # +2 because we skip first two values
            sequence.append(sequence[-1] + sequence[-2])
        
        # Return sequence starting from the third element (skipping 0, 1)
        return sequence[:n]
    else:
        # Standard Fibonacci ratios: 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...
        sequence = [1, 2, 3]
        while len(sequence) < n:
            sequence.append(sequence[-1] + sequence[-2])
        
        return sequence[:n]


def fibonacci_ratios(extended: bool = False) -> List[float]:
    """
    Get standard Fibonacci ratios used in technical analysis.
    
    Args:
        extended: Whether to include extended ratios
            
    Returns:
        List of Fibonacci ratios
    """
    standard_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    if extended:
        # Include extension ratios
        extended_ratios = [1.272, 1.382, 1.618, 2.0, 2.618, 3.618, 4.236]
        return standard_ratios + extended_ratios
    else:
        return standard_ratios


def calculate_fibonacci_retracement_levels(
    start_price: float, 
    end_price: float, 
    levels: List[float] = None
) -> Dict[float, float]:
    """
    Calculate Fibonacci retracement levels.
    
    Args:
        start_price: Starting price point
        end_price: Ending price point
        levels: List of Fibonacci ratios to calculate
            
    Returns:
        Dictionary mapping Fibonacci ratios to price levels
    """
    if levels is None:
        levels = fibonacci_ratios()
    
    price_range = end_price - start_price
    result = {}
    
    for level in levels:
        level_price = end_price - (price_range * level)
        result[level] = level_price
    
    return result


def calculate_fibonacci_extension_levels(
    start_price: float, 
    end_price: float, 
    retracement_price: float,
    levels: List[float] = None
) -> Dict[float, float]:
    """
    Calculate Fibonacci extension levels.
    
    Args:
        start_price: Starting price point
        end_price: Ending price point
        retracement_price: Retracement price point
        levels: List of Fibonacci ratios to calculate
            
    Returns:
        Dictionary mapping Fibonacci ratios to price levels
    """
    if levels is None:
        levels = fibonacci_ratios(extended=True)
    
    price_range = end_price - start_price
    result = {}
    
    for level in levels:
        level_price = retracement_price + (price_range * level)
        result[level] = level_price
    
    return result


def format_fibonacci_level(level: float) -> str:
    """
    Format a Fibonacci level for use in column names.
    
    Args:
        level: Fibonacci level
            
    Returns:
        Formatted level string
    """
    return str(level).replace('.', '_')


def is_golden_ratio(ratio: float, tolerance: float = 0.05) -> bool:
    """
    Check if a ratio is close to the golden ratio (0.618 or 1.618).
    
    Args:
        ratio: Ratio to check
        tolerance: Tolerance for comparison
            
    Returns:
        True if the ratio is close to the golden ratio
    """
    return (abs(ratio - 0.618) <= tolerance or 
            abs(ratio - 1.618) <= tolerance)


def is_fibonacci_ratio(ratio: float, tolerance: float = 0.05) -> bool:
    """
    Check if a ratio is close to any Fibonacci ratio.
    
    Args:
        ratio: Ratio to check
        tolerance: Tolerance for comparison
            
    Returns:
        True if the ratio is close to any Fibonacci ratio
    """
    fib_ratios = fibonacci_ratios(extended=True)
    
    for fib_ratio in fib_ratios:
        if abs(ratio - fib_ratio) <= tolerance:
            return True
    
    return False