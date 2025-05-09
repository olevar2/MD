"""
Statistical Utilities

This module provides statistical utilities for feedback analysis.
"""

import logging
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_basic_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of values.
    
    Args:
        values: List of values
        
    Returns:
        Dict[str, float]: Dictionary with statistics
    """
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std_dev": 0.0,
            "variance": 0.0
        }
    
    try:
        # Use numpy for efficient calculations
        values_array = np.array(values)
        
        return {
            "count": len(values),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "mean": float(np.mean(values_array)),
            "median": float(np.median(values_array)),
            "std_dev": float(np.std(values_array)),
            "variance": float(np.var(values_array))
        }
    except:
        # Fallback to manual calculation
        count = len(values)
        min_val = min(values)
        max_val = max(values)
        mean = sum(values) / count
        
        # Sort for median
        sorted_values = sorted(values)
        if count % 2 == 0:
            median = (sorted_values[count // 2 - 1] + sorted_values[count // 2]) / 2
        else:
            median = sorted_values[count // 2]
        
        # Calculate variance and std_dev
        variance = sum((x - mean) ** 2 for x in values) / count
        std_dev = math.sqrt(variance)
        
        return {
            "count": count,
            "min": min_val,
            "max": max_val,
            "mean": mean,
            "median": median,
            "std_dev": std_dev,
            "variance": variance
        }


def calculate_percentiles(values: List[float], percentiles: List[int] = None) -> Dict[str, float]:
    """
    Calculate percentiles for a list of values.
    
    Args:
        values: List of values
        percentiles: List of percentiles to calculate (default: [25, 50, 75, 90, 95, 99])
        
    Returns:
        Dict[str, float]: Dictionary with percentiles
    """
    if not values:
        return {}
    
    if percentiles is None:
        percentiles = [25, 50, 75, 90, 95, 99]
    
    try:
        # Use numpy for efficient calculations
        values_array = np.array(values)
        
        result = {}
        for p in percentiles:
            result[f"p{p}"] = float(np.percentile(values_array, p))
        
        return result
    except:
        # Fallback to manual calculation
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        result = {}
        for p in percentiles:
            index = int(count * p / 100)
            if index >= count:
                index = count - 1
            result[f"p{p}"] = sorted_values[index]
        
        return result


def calculate_moving_average(
    values: List[float], window_size: int = 5
) -> List[float]:
    """
    Calculate moving average for a list of values.
    
    Args:
        values: List of values
        window_size: Size of the moving window
        
    Returns:
        List[float]: Moving averages
    """
    if not values or window_size <= 0:
        return []
    
    try:
        # Use numpy for efficient calculations
        values_array = np.array(values)
        result = []
        
        for i in range(len(values)):
            start = max(0, i - window_size + 1)
            window = values_array[start:i+1]
            result.append(float(np.mean(window)))
        
        return result
    except:
        # Fallback to manual calculation
        result = []
        
        for i in range(len(values)):
            start = max(0, i - window_size + 1)
            window = values[start:i+1]
            result.append(sum(window) / len(window))
        
        return result


def calculate_correlation(values1: List[float], values2: List[float]) -> Tuple[float, str]:
    """
    Calculate correlation coefficient and significance.
    
    Args:
        values1: First list of values
        values2: Second list of values
        
    Returns:
        Tuple[float, str]: Correlation coefficient and significance
    """
    if not values1 or not values2:
        return 0.0, "none"
    
    # Ensure equal length
    min_len = min(len(values1), len(values2))
    values1 = values1[:min_len]
    values2 = values2[:min_len]
    
    try:
        # Use numpy for efficient calculations
        correlation = float(np.corrcoef(values1, values2)[0, 1])
    except:
        # Fallback to manual calculation
        mean1 = sum(values1) / len(values1)
        mean2 = sum(values2) / len(values2)
        
        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(values1, values2))
        denom1 = sum((x - mean1) ** 2 for x in values1) ** 0.5
        denom2 = sum((y - mean2) ** 2 for y in values2) ** 0.5
        
        if denom1 == 0 or denom2 == 0:
            correlation = 0.0
        else:
            correlation = numerator / (denom1 * denom2)
    
    # Determine significance
    abs_corr = abs(correlation)
    if abs_corr < 0.3:
        significance = "weak"
    elif abs_corr < 0.7:
        significance = "moderate"
    else:
        significance = "strong"
    
    return correlation, significance


def detect_outliers(
    values: List[float], method: str = "zscore", threshold: float = 3.0
) -> List[int]:
    """
    Detect outliers in a list of values.
    
    Args:
        values: List of values
        method: Method for outlier detection ("zscore" or "iqr")
        threshold: Threshold for outlier detection
        
    Returns:
        List[int]: Indices of outliers
    """
    if not values:
        return []
    
    try:
        values_array = np.array(values)
        
        if method == "zscore":
            # Z-score method
            mean = np.mean(values_array)
            std_dev = np.std(values_array)
            
            if std_dev == 0:
                return []
            
            z_scores = np.abs((values_array - mean) / std_dev)
            outliers = np.where(z_scores > threshold)[0].tolist()
            
        elif method == "iqr":
            # IQR method
            q1 = np.percentile(values_array, 25)
            q3 = np.percentile(values_array, 75)
            iqr = q3 - q1
            
            if iqr == 0:
                return []
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outliers = np.where((values_array < lower_bound) | (values_array > upper_bound))[0].tolist()
            
        else:
            logger.warning(f"Unknown outlier detection method: {method}")
            return []
        
        return outliers
    except:
        # Fallback to manual calculation
        outliers = []
        
        if method == "zscore":
            # Z-score method
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std_dev = math.sqrt(variance)
            
            if std_dev == 0:
                return []
            
            for i, value in enumerate(values):
                z_score = abs((value - mean) / std_dev)
                if z_score > threshold:
                    outliers.append(i)
                    
        elif method == "iqr":
            # IQR method
            sorted_values = sorted(values)
            n = len(sorted_values)
            
            q1_idx = n // 4
            q3_idx = (3 * n) // 4
            
            q1 = sorted_values[q1_idx]
            q3 = sorted_values[q3_idx]
            iqr = q3 - q1
            
            if iqr == 0:
                return []
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            for i, value in enumerate(values):
                if value < lower_bound or value > upper_bound:
                    outliers.append(i)
        
        return outliers