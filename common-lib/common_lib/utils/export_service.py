"""
Export Service Module

This module provides utilities for exporting data to various formats.
It supports exporting data to CSV, JSON, Parquet, Excel, and other formats.
"""

import io
import csv
import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable

import pandas as pd
import numpy as np

from common_lib.schemas import OHLCVData

logger = logging.getLogger(__name__)


def convert_to_csv(
    data: Union[List[Dict[str, Any]], List[OHLCVData], pd.DataFrame],
    include_header: bool = True,
    delimiter: str = ',',
    custom_formatter: Optional[Callable[[Any], str]] = None
) -> str:
    """
    Convert data to CSV format.
    
    Args:
        data: Data to convert (list of dictionaries, list of OHLCVData, or DataFrame)
        include_header: Whether to include header row
        delimiter: Delimiter character
        custom_formatter: Optional function to format values
        
    Returns:
        String containing CSV data
    """
    if not data:
        return ""
    
    # Create output buffer
    output = io.StringIO()
    
    # Convert data to DataFrame if needed
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data[0], OHLCVData):
        # Convert OHLCVData to list of dictionaries
        df = pd.DataFrame([
            {
                "timestamp": point.timestamp.isoformat(),
                "open": point.open,
                "high": point.high,
                "low": point.low,
                "close": point.close,
                "volume": point.volume
            }
            for point in data
        ])
    else:
        # Assume list of dictionaries
        df = pd.DataFrame(data)
    
    # Apply custom formatter if provided
    if custom_formatter:
        df = df.applymap(lambda x: custom_formatter(x) if x is not None else "")
    
    # Write to CSV
    df.to_csv(output, index=False, header=include_header, sep=delimiter)
    
    # Get string value and rewind buffer
    output.seek(0)
    return output.read()


def convert_to_json(
    data: Union[List[Dict[str, Any]], List[OHLCVData], pd.DataFrame],
    orient: str = 'records',
    date_format: str = 'iso',
    indent: Optional[int] = None,
    custom_formatter: Optional[Callable[[Any], Any]] = None
) -> str:
    """
    Convert data to JSON format.
    
    Args:
        data: Data to convert (list of dictionaries, list of OHLCVData, or DataFrame)
        orient: JSON orientation (records, columns, index, split, table)
        date_format: Date format (iso, epoch)
        indent: Indentation level
        custom_formatter: Optional function to format values
        
    Returns:
        String containing JSON data
    """
    if not data:
        return "[]"
    
    # Convert data to appropriate format
    if isinstance(data, pd.DataFrame):
        # Use pandas to_json for DataFrames
        return data.to_json(orient=orient, date_format=date_format, indent=indent)
    elif isinstance(data[0], OHLCVData):
        # Convert OHLCVData to list of dictionaries
        records = [
            {
                "timestamp": point.timestamp.isoformat(),
                "open": point.open,
                "high": point.high,
                "low": point.low,
                "close": point.close,
                "volume": point.volume
            }
            for point in data
        ]
    else:
        # Assume list of dictionaries
        records = data
    
    # Apply custom formatter if provided
    if custom_formatter:
        def apply_formatter(obj):
    """
    Apply formatter.
    
    Args:
        obj: Description of obj
    
    """

            if isinstance(obj, dict):
                return {k: apply_formatter(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [apply_formatter(item) for item in obj]
            else:
                return custom_formatter(obj)
        
        records = apply_formatter(records)
    
    # Convert to JSON
    return json.dumps(records, indent=indent, default=str)


def convert_to_parquet(
    data: Union[List[Dict[str, Any]], List[OHLCVData], pd.DataFrame],
    compression: str = 'snappy',
    custom_formatter: Optional[Callable[[Any], Any]] = None
) -> bytes:
    """
    Convert data to Parquet format.
    
    Args:
        data: Data to convert (list of dictionaries, list of OHLCVData, or DataFrame)
        compression: Compression algorithm (snappy, gzip, brotli, none)
        custom_formatter: Optional function to format values
        
    Returns:
        Bytes containing Parquet data
    """
    if not data:
        # Return empty DataFrame as Parquet
        return pd.DataFrame().to_parquet(compression=compression)
    
    # Convert data to DataFrame if needed
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data[0], OHLCVData):
        # Convert OHLCVData to list of dictionaries
        df = pd.DataFrame([
            {
                "timestamp": point.timestamp,
                "open": point.open,
                "high": point.high,
                "low": point.low,
                "close": point.close,
                "volume": point.volume
            }
            for point in data
        ])
    else:
        # Assume list of dictionaries
        df = pd.DataFrame(data)
    
    # Apply custom formatter if provided
    if custom_formatter:
        df = df.applymap(lambda x: custom_formatter(x) if x is not None else None)
    
    # Convert to Parquet
    buffer = io.BytesIO()
    df.to_parquet(buffer, compression=compression)
    buffer.seek(0)
    return buffer.getvalue()


def convert_to_excel(
    data: Union[List[Dict[str, Any]], List[OHLCVData], pd.DataFrame],
    sheet_name: str = 'Sheet1',
    custom_formatter: Optional[Callable[[Any], Any]] = None
) -> bytes:
    """
    Convert data to Excel format.
    
    Args:
        data: Data to convert (list of dictionaries, list of OHLCVData, or DataFrame)
        sheet_name: Name of the Excel sheet
        custom_formatter: Optional function to format values
        
    Returns:
        Bytes containing Excel data
    """
    if not data:
        # Return empty DataFrame as Excel
        return pd.DataFrame().to_excel(sheet_name=sheet_name)
    
    # Convert data to DataFrame if needed
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data[0], OHLCVData):
        # Convert OHLCVData to list of dictionaries
        df = pd.DataFrame([
            {
                "timestamp": point.timestamp,
                "open": point.open,
                "high": point.high,
                "low": point.low,
                "close": point.close,
                "volume": point.volume
            }
            for point in data
        ])
    else:
        # Assume list of dictionaries
        df = pd.DataFrame(data)
    
    # Apply custom formatter if provided
    if custom_formatter:
        df = df.applymap(lambda x: custom_formatter(x) if x is not None else None)
    
    # Convert to Excel
    buffer = io.BytesIO()
    df.to_excel(buffer, sheet_name=sheet_name, index=False)
    buffer.seek(0)
    return buffer.getvalue()


def format_ohlcv_for_json(data: List[OHLCVData]) -> List[Dict[str, Any]]:
    """
    Format OHLCV data for JSON serialization.
    
    Args:
        data: List of OHLCV data points
        
    Returns:
        List of dictionaries with formatted OHLCV data
    """
    return [
        {
            "timestamp": point.timestamp.isoformat(),
            "open": float(point.open),
            "high": float(point.high),
            "low": float(point.low),
            "close": float(point.close),
            "volume": float(point.volume)
        }
        for point in data
    ]


def format_timestamp(timestamp: Any) -> str:
    """
    Format timestamp for export.
    
    Args:
        timestamp: Timestamp to format
        
    Returns:
        Formatted timestamp string
    """
    if pd.isna(timestamp):
        return ""
    
    if isinstance(timestamp, pd.Timestamp):
        return timestamp.isoformat()
    
    if isinstance(timestamp, (int, float)):
        # Assume Unix timestamp
        return pd.Timestamp(timestamp, unit='s').isoformat()
    
    # Try to convert to string
    return str(timestamp)


def format_numeric(value: Any, precision: int = 8) -> str:
    """
    Format numeric value for export.
    
    Args:
        value: Numeric value to format
        precision: Decimal precision
        
    Returns:
        Formatted numeric string
    """
    if pd.isna(value):
        return ""
    
    if isinstance(value, (int, float, np.number)):
        if isinstance(value, (int, np.integer)):
            return str(value)
        else:
            return f"{value:.{precision}f}".rstrip('0').rstrip('.') if '.' in f"{value:.{precision}f}" else f"{value:.{precision}f}"
    
    # Try to convert to string
    return str(value)
