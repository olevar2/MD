"""
Export service module.

This module provides functionality for...
"""

import io
import csv
import json
import pandas as pd
from typing import List, Dict, Any

from ..models.schemas import OHLCVData


def convert_to_csv(data: List[OHLCVData]) -> str:
    """
    Convert a list of OHLCV data to CSV format.
    
    Args:
        data: List of OHLCV data points
        
    Returns:
        String containing CSV data
    """
    if not data:
        return ""
        
    # Create output buffer
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
    
    # Write data rows
    for point in data:
        writer.writerow([
            point.timestamp.isoformat(),
            point.open,
            point.high,
            point.low,
            point.close,
            point.volume
        ])
    
    # Get string value and rewind buffer
    output.seek(0)
    return output.read()


def convert_to_parquet(data: List[OHLCVData]) -> bytes:
    """
    Convert a list of OHLCV data to Parquet format.
    
    Args:
        data: List of OHLCV data points
        
    Returns:
        Bytes containing Parquet data
    """
    if not data:
        return bytes()
        
    # Convert to pandas DataFrame
    df = pd.DataFrame([d.dict() for d in data])
    
    # Convert timestamp column to pandas datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create output buffer
    buffer = io.BytesIO()
    
    # Write to parquet format
    df.to_parquet(buffer, index=False, compression="snappy")
    
    # Get bytes value and rewind buffer
    buffer.seek(0)
    return buffer.getvalue()


def format_ohlcv_for_json(data: List[OHLCVData]) -> List[Dict[str, Any]]:
    """
    Format OHLCV data for JSON serialization.
    
    Args:
        data: List of OHLCV data points
        
    Returns:
        List of dictionaries ready for JSON serialization
    """
    return [
        {
            "timestamp": d.timestamp.isoformat(),
            "open": d.open,
            "high": d.high,
            "low": d.low,
            "close": d.close,
            "volume": d.volume
        }
        for d in data
    ]