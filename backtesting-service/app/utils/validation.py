"""
Validation Utilities

This module provides validation functions for backtesting.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def validate_backtest_request(request: Dict[str, Any]) -> None:
    """
    Validate a backtest request.
    
    Args:
        request: Dictionary containing request parameters
        
    Raises:
        ValueError: If request is invalid
    """
    required_fields = ['strategy_id', 'symbol', 'timeframe', 'start_date', 'end_date']
    for field in required_fields:
        if field not in request:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate timeframe
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
    if request['timeframe'] not in valid_timeframes:
        raise ValueError(f"Invalid timeframe: {request['timeframe']}. Must be one of {valid_timeframes}")
    
    # Validate dates
    start_date = request['start_date']
    end_date = request['end_date']
    
    if isinstance(start_date, str):
        try:
            start_date = datetime.fromisoformat(start_date)
        except ValueError:
            raise ValueError(f"Invalid start_date format: {start_date}")
    
    if isinstance(end_date, str):
        try:
            end_date = datetime.fromisoformat(end_date)
        except ValueError:
            raise ValueError(f"Invalid end_date format: {end_date}")
    
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date")
    
    # Validate initial_balance if provided
    if 'initial_balance' in request and request['initial_balance'] <= 0:
        raise ValueError("initial_balance must be greater than 0")

def validate_optimization_request(request: Dict[str, Any]) -> None:
    """
    Validate an optimization request.
    
    Args:
        request: Dictionary containing request parameters
        
    Raises:
        ValueError: If request is invalid
    """
    required_fields = ['strategy_id', 'symbol', 'timeframe', 'start_date', 'end_date', 'parameters_to_optimize']
    for field in required_fields:
        if field not in request:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate timeframe
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
    if request['timeframe'] not in valid_timeframes:
        raise ValueError(f"Invalid timeframe: {request['timeframe']}. Must be one of {valid_timeframes}")
    
    # Validate dates
    start_date = request['start_date']
    end_date = request['end_date']
    
    if isinstance(start_date, str):
        try:
            start_date = datetime.fromisoformat(start_date)
        except ValueError:
            raise ValueError(f"Invalid start_date format: {start_date}")
    
    if isinstance(end_date, str):
        try:
            end_date = datetime.fromisoformat(end_date)
        except ValueError:
            raise ValueError(f"Invalid end_date format: {end_date}")
    
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date")
    
    # Validate initial_balance if provided
    if 'initial_balance' in request and request['initial_balance'] <= 0:
        raise ValueError("initial_balance must be greater than 0")
    
    # Validate parameters_to_optimize
    if not isinstance(request['parameters_to_optimize'], dict) or not request['parameters_to_optimize']:
        raise ValueError("parameters_to_optimize must be a non-empty dictionary")
    
    for param, param_range in request['parameters_to_optimize'].items():
        if not isinstance(param_range, dict):
            raise ValueError(f"Parameter range for {param} must be a dictionary")
        
        if 'min' not in param_range or 'max' not in param_range:
            raise ValueError(f"Parameter range for {param} must contain 'min' and 'max' keys")
        
        if param_range['min'] > param_range['max']:
            raise ValueError(f"Parameter range for {param}: min cannot be greater than max")
    
    # Validate optimization_metric if provided
    valid_metrics = ['sharpe_ratio', 'total_return', 'profit_factor', 'win_rate', 'max_drawdown']
    if 'optimization_metric' in request and request['optimization_metric'] not in valid_metrics:
        raise ValueError(f"Invalid optimization_metric: {request['optimization_metric']}. Must be one of {valid_metrics}")
    
    # Validate optimization_method if provided
    valid_methods = ['grid_search', 'random_search', 'bayesian_optimization']
    if 'optimization_method' in request and request['optimization_method'] not in valid_methods:
        raise ValueError(f"Invalid optimization_method: {request['optimization_method']}. Must be one of {valid_methods}")

def validate_walk_forward_test_request(request: Dict[str, Any]) -> None:
    """
    Validate a walk-forward test request.
    
    Args:
        request: Dictionary containing request parameters
        
    Raises:
        ValueError: If request is invalid
    """
    required_fields = ['strategy_id', 'symbol', 'timeframe', 'start_date', 'end_date', 'optimization_window', 'test_window', 'parameters_to_optimize']
    for field in required_fields:
        if field not in request:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate timeframe
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
    if request['timeframe'] not in valid_timeframes:
        raise ValueError(f"Invalid timeframe: {request['timeframe']}. Must be one of {valid_timeframes}")
    
    # Validate dates
    start_date = request['start_date']
    end_date = request['end_date']
    
    if isinstance(start_date, str):
        try:
            start_date = datetime.fromisoformat(start_date)
        except ValueError:
            raise ValueError(f"Invalid start_date format: {start_date}")
    
    if isinstance(end_date, str):
        try:
            end_date = datetime.fromisoformat(end_date)
        except ValueError:
            raise ValueError(f"Invalid end_date format: {end_date}")
    
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date")
    
    # Validate initial_balance if provided
    if 'initial_balance' in request and request['initial_balance'] <= 0:
        raise ValueError("initial_balance must be greater than 0")
    
    # Validate optimization_window and test_window
    if request['optimization_window'] <= 0:
        raise ValueError("optimization_window must be greater than 0")
    
    if request['test_window'] <= 0:
        raise ValueError("test_window must be greater than 0")
    
    # Validate parameters_to_optimize
    if not isinstance(request['parameters_to_optimize'], dict) or not request['parameters_to_optimize']:
        raise ValueError("parameters_to_optimize must be a non-empty dictionary")
    
    for param, param_range in request['parameters_to_optimize'].items():
        if not isinstance(param_range, dict):
            raise ValueError(f"Parameter range for {param} must be a dictionary")
        
        if 'min' not in param_range or 'max' not in param_range:
            raise ValueError(f"Parameter range for {param} must contain 'min' and 'max' keys")
        
        if param_range['min'] > param_range['max']:
            raise ValueError(f"Parameter range for {param}: min cannot be greater than max")
    
    # Validate optimization_metric if provided
    valid_metrics = ['sharpe_ratio', 'total_return', 'profit_factor', 'win_rate', 'max_drawdown']
    if 'optimization_metric' in request and request['optimization_metric'] not in valid_metrics:
        raise ValueError(f"Invalid optimization_metric: {request['optimization_metric']}. Must be one of {valid_metrics}")

def validate_market_data(data: pd.DataFrame) -> None:
    """
    Validate market data for backtesting.
    
    Args:
        data: DataFrame containing market data
        
    Raises:
        ValueError: If data is invalid for backtesting
    """
    # Check if data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    
    # Check if data is empty
    if data.empty:
        raise ValueError("Data cannot be empty")
    
    # Check if data has required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Data must have a '{column}' column")
    
    # Check if data has a datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have a datetime index")
    
    # Check for missing values
    missing_values = data[required_columns].isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Data contains missing values: {missing_values}")
    
    # Check for duplicate indices
    if data.index.duplicated().any():
        raise ValueError("Data contains duplicate timestamps")
    
    # Check for price anomalies
    if (data['high'] < data['low']).any():
        raise ValueError("Data contains anomalies: high price is less than low price")
    
    if (data['close'] > data['high']).any() or (data['close'] < data['low']).any():
        raise ValueError("Data contains anomalies: close price is outside high-low range")
    
    if (data['open'] > data['high']).any() or (data['open'] < data['low']).any():
        raise ValueError("Data contains anomalies: open price is outside high-low range")
    
    # Check for negative prices or volume
    if (data[['open', 'high', 'low', 'close']] <= 0).any().any():
        raise ValueError("Data contains non-positive prices")
    
    if (data['volume'] < 0).any():
        raise ValueError("Data contains negative volume")
