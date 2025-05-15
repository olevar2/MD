"""
Data processing utilities for Market Analysis Service.

This module provides utilities for processing market data.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

def process_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process market data for analysis.
    
    Args:
        data: Market data
        
    Returns:
        Processed market data
    """
    # Make a copy to avoid modifying the input data
    processed_data = data.copy()
    
    # Ensure required columns exist
    required_columns = ["open", "high", "low", "close"]
    
    for column in required_columns:
        if column not in processed_data.columns:
            logger.error(f"Required column {column} not found in market data")
            raise ValueError(f"Required column {column} not found in market data")
            
    # Handle missing values
    processed_data = processed_data.fillna(method="ffill")
    
    # Ensure data is sorted by timestamp
    if "timestamp" in processed_data.columns:
        processed_data = processed_data.sort_values("timestamp")
        
    return processed_data
    
def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for market data.
    
    Args:
        data: Market data
        
    Returns:
        Market data with technical indicators
    """
    # Make a copy to avoid modifying the input data
    processed_data = data.copy()
    
    # Ensure required columns exist
    required_columns = ["open", "high", "low", "close"]
    
    for column in required_columns:
        if column not in processed_data.columns:
            logger.error(f"Required column {column} not found in market data")
            raise ValueError(f"Required column {column} not found in market data")
            
    # Calculate SMA (Simple Moving Average)
    processed_data["sma_20"] = processed_data["close"].rolling(window=20).mean()
    processed_data["sma_50"] = processed_data["close"].rolling(window=50).mean()
    processed_data["sma_200"] = processed_data["close"].rolling(window=200).mean()
    
    # Calculate EMA (Exponential Moving Average)
    processed_data["ema_12"] = processed_data["close"].ewm(span=12).mean()
    processed_data["ema_26"] = processed_data["close"].ewm(span=26).mean()
    
    # Calculate MACD (Moving Average Convergence Divergence)
    processed_data["macd"] = processed_data["ema_12"] - processed_data["ema_26"]
    processed_data["macd_signal"] = processed_data["macd"].ewm(span=9).mean()
    processed_data["macd_histogram"] = processed_data["macd"] - processed_data["macd_signal"]
    
    # Calculate RSI (Relative Strength Index)
    delta = processed_data["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    
    rs = gain / loss
    processed_data["rsi_14"] = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    processed_data["bb_middle"] = processed_data["close"].rolling(window=20).mean()
    processed_data["bb_std"] = processed_data["close"].rolling(window=20).std()
    processed_data["bb_upper"] = processed_data["bb_middle"] + 2 * processed_data["bb_std"]
    processed_data["bb_lower"] = processed_data["bb_middle"] - 2 * processed_data["bb_std"]
    
    # Calculate ATR (Average True Range)
    tr1 = processed_data["high"] - processed_data["low"]
    tr2 = abs(processed_data["high"] - processed_data["close"].shift())
    tr3 = abs(processed_data["low"] - processed_data["close"].shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    processed_data["atr_14"] = tr.rolling(window=14).mean()
    
    return processed_data
    
def resample_market_data(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample market data to a different timeframe.
    
    Args:
        data: Market data
        timeframe: Target timeframe
        
    Returns:
        Resampled market data
    """
    # Ensure timestamp column exists and is datetime
    if "timestamp" not in data.columns:
        logger.error("Timestamp column not found in market data")
        raise ValueError("Timestamp column not found in market data")
        
    # Make a copy to avoid modifying the input data
    processed_data = data.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(processed_data["timestamp"]):
        processed_data["timestamp"] = pd.to_datetime(processed_data["timestamp"])
        
    # Set timestamp as index
    processed_data = processed_data.set_index("timestamp")
    
    # Map timeframe to pandas frequency
    timeframe_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D",
        "1w": "1W",
        "1M": "1M"
    }
    
    if timeframe not in timeframe_map:
        logger.error(f"Unsupported timeframe: {timeframe}")
        raise ValueError(f"Unsupported timeframe: {timeframe}")
        
    freq = timeframe_map[timeframe]
    
    # Resample data
    resampled = processed_data.resample(freq).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum" if "volume" in processed_data.columns else None
    })
    
    # Reset index
    resampled = resampled.reset_index()
    
    return resampled