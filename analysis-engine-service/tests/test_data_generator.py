"""
Test Data Generator

This module provides utilities for generating test data for the Analysis Engine.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

def generate_test_data(
    symbols: List[str],
    timeframes: List[str],
    days: int = 30,
    volatility: float = 0.01,
    trend: float = 0.0001
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Generate synthetic price data for testing.
    
    Args:
        symbols: List of currency pairs
        timeframes: List of timeframes
        days: Number of days of data to generate
        volatility: Volatility parameter for price generation
        trend: Trend parameter for price generation
        
    Returns:
        Dictionary mapping symbols to dictionaries mapping timeframes to DataFrames
    """
    # Base prices for different pairs
    base_prices = {
        "EURUSD": 1.1000,
        "GBPUSD": 1.3000,
        "USDJPY": 110.00,
        "AUDUSD": 0.7000,
        "USDCAD": 1.3000,
        "EURGBP": 0.8500,
        "EURJPY": 130.00,
        "GBPJPY": 150.00
    }
    
    # Map timeframe to timedelta
    timeframe_map = {
        "M1": timedelta(minutes=1),
        "M5": timedelta(minutes=5),
        "M15": timedelta(minutes=15),
        "M30": timedelta(minutes=30),
        "H1": timedelta(hours=1),
        "H4": timedelta(hours=4),
        "D1": timedelta(days=1)
    }
    
    # Generate data for each symbol and timeframe
    result = {}
    
    for symbol in symbols:
        result[symbol] = {}
        
        # Get base price for the symbol
        base_price = base_prices.get(symbol, 1.0)
        
        for timeframe in timeframes:
            # Get timedelta for the timeframe
            delta = timeframe_map.get(timeframe)
            if delta is None:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            # Generate timestamps
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            timestamps = []
            current = start_date
            while current <= end_date:
                timestamps.append(current)
                current += delta
            
            # Set random seed based on symbol
            np.random.seed(hash(symbol) % 2**32)
            
            # Generate random walk
            returns = np.random.normal(trend, volatility, len(timestamps))
            cumulative_returns = np.cumsum(returns)
            
            # Add some cyclical patterns
            t = np.linspace(0, 10 * np.pi, len(timestamps))
            cycles = 0.005 * np.sin(t) + 0.003 * np.sin(2 * t) + 0.002 * np.sin(3 * t)
            
            # Combine components
            close_prices = base_price * np.exp(cumulative_returns + cycles)
            
            # Generate OHLC data
            high_prices = close_prices * np.exp(np.random.uniform(0, volatility, len(timestamps)))
            low_prices = close_prices * np.exp(-np.random.uniform(0, volatility, len(timestamps)))
            open_prices = low_prices + np.random.uniform(0, 1, len(timestamps)) * (high_prices - low_prices)
            
            # Generate volume data
            volume = np.random.uniform(1000, 10000, len(timestamps))
            
            # Create DataFrame
            df = pd.DataFrame({
                "timestamp": timestamps,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volume
            })
            
            result[symbol][timeframe] = df
    
    return result

def generate_pattern_labels(
    price_data: Dict[str, Dict[str, pd.DataFrame]],
    num_patterns: int = 8
) -> Dict[str, List[List[int]]]:
    """
    Generate pattern labels for testing.
    
    Args:
        price_data: Dictionary mapping symbols to dictionaries mapping timeframes to DataFrames
        num_patterns: Number of patterns to generate labels for
        
    Returns:
        Dictionary mapping symbols to lists of pattern labels
    """
    # Pattern names
    pattern_names = [
        "double_top",
        "double_bottom",
        "head_and_shoulders",
        "inverse_head_and_shoulders",
        "triangle",
        "wedge",
        "rectangle",
        "flag"
    ][:num_patterns]
    
    # Generate labels for each symbol
    result = {}
    
    for symbol, timeframe_data in price_data.items():
        # Use H1 timeframe for pattern labels
        if "H1" not in timeframe_data:
            continue
        
        df = timeframe_data["H1"]
        
        # Generate random labels
        np.random.seed(hash(symbol) % 2**32)
        
        # Create windows
        window_size = 30
        num_windows = len(df) - window_size + 1
        
        # Generate labels for each window
        labels = []
        for i in range(num_windows):
            # Generate random label
            label = [0] * num_patterns
            
            # 10% chance of having a pattern
            if np.random.random() < 0.1:
                # Select a random pattern
                pattern_index = np.random.randint(0, num_patterns)
                label[pattern_index] = 1
            
            labels.append(label)
        
        result[symbol] = labels
    
    return result

def generate_correlation_matrix(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Generate a correlation matrix for testing.
    
    Args:
        symbols: List of currency pairs
        
    Returns:
        Dictionary mapping symbols to dictionaries mapping symbols to correlations
    """
    # Generate random correlation matrix
    np.random.seed(42)
    
    # Create empty result
    result = {}
    
    for i, symbol1 in enumerate(symbols):
        result[symbol1] = {}
        
        for j, symbol2 in enumerate(symbols):
            if i == j:
                # Self-correlation is 1.0
                result[symbol1][symbol2] = 1.0
            elif j > i:
                # Generate random correlation
                correlation = np.random.uniform(-0.9, 0.9)
                result[symbol1][symbol2] = correlation
                
                # Ensure symmetry
                if symbol2 not in result:
                    result[symbol2] = {}
                result[symbol2][symbol1] = correlation
    
    return result
