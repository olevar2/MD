"""
Validation Utilities

This module provides validation functions for causal analysis.
"""
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def validate_data_for_causal_analysis(data: pd.DataFrame) -> None:
    """
    Validate data for causal analysis.
    
    Args:
        data: DataFrame containing time series data
        
    Raises:
        ValueError: If data is invalid for causal analysis
    """
    # Check if data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    
    # Check if data is empty
    if data.empty:
        raise ValueError("Data cannot be empty")
    
    # Check if data has at least 2 columns
    if len(data.columns) < 2:
        raise ValueError("Data must have at least 2 columns for causal analysis")
    
    # Check if data has numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) < 2:
        raise ValueError("Data must have at least 2 numeric columns for causal analysis")
    
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Data contains missing values: {missing_values}")
    
    # Check for constant columns
    constant_columns = [col for col in data.columns if data[col].nunique() == 1]
    if constant_columns:
        logger.warning(f"Data contains constant columns: {constant_columns}")
    
    # Check for sufficient data points
    min_rows = 30  # Minimum number of data points for meaningful causal analysis
    if len(data) < min_rows:
        logger.warning(f"Data contains only {len(data)} rows, which may be insufficient for reliable causal analysis")


def validate_causal_graph_request(request: Dict[str, Any]) -> None:
    """
    Validate a causal graph request.
    
    Args:
        request: Dictionary containing request parameters
        
    Raises:
        ValueError: If request is invalid
    """
    required_fields = ['symbol', 'timeframe', 'start_date']
    for field in required_fields:
        if field not in request:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate algorithm
    valid_algorithms = ['granger', 'pc', 'dowhy']
    algorithm = request.get('algorithm', 'granger')
    if algorithm not in valid_algorithms:
        raise ValueError(f"Invalid algorithm: {algorithm}. Must be one of {valid_algorithms}")
    
    # Validate timeframe
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
    if request['timeframe'] not in valid_timeframes:
        raise ValueError(f"Invalid timeframe: {request['timeframe']}. Must be one of {valid_timeframes}")
    
    # Validate dates
    start_date = request['start_date']
    end_date = request.get('end_date')
    
    if end_date and start_date > end_date:
        raise ValueError("start_date cannot be after end_date")


def validate_intervention_effect_request(request: Dict[str, Any]) -> None:
    """
    Validate an intervention effect request.
    
    Args:
        request: Dictionary containing request parameters
        
    Raises:
        ValueError: If request is invalid
    """
    required_fields = ['symbol', 'timeframe', 'start_date', 'treatment', 'outcome']
    for field in required_fields:
        if field not in request:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate algorithm
    valid_algorithms = ['dowhy', 'causalml']
    algorithm = request.get('algorithm', 'dowhy')
    if algorithm not in valid_algorithms:
        raise ValueError(f"Invalid algorithm: {algorithm}. Must be one of {valid_algorithms}")
    
    # Validate timeframe
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
    if request['timeframe'] not in valid_timeframes:
        raise ValueError(f"Invalid timeframe: {request['timeframe']}. Must be one of {valid_timeframes}")
    
    # Validate dates
    start_date = request['start_date']
    end_date = request.get('end_date')
    
    if end_date and start_date > end_date:
        raise ValueError("start_date cannot be after end_date")
    
    # Validate treatment and outcome are different
    if request['treatment'] == request['outcome']:
        raise ValueError("treatment and outcome cannot be the same variable")


def validate_counterfactual_request(request: Dict[str, Any]) -> None:
    """
    Validate a counterfactual request.
    
    Args:
        request: Dictionary containing request parameters
        
    Raises:
        ValueError: If request is invalid
    """
    required_fields = ['symbol', 'timeframe', 'start_date', 'intervention', 'target_variables']
    for field in required_fields:
        if field not in request:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate algorithm
    valid_algorithms = ['counterfactual', 'causalml']
    algorithm = request.get('algorithm', 'counterfactual')
    if algorithm not in valid_algorithms:
        raise ValueError(f"Invalid algorithm: {algorithm}. Must be one of {valid_algorithms}")
    
    # Validate timeframe
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
    if request['timeframe'] not in valid_timeframes:
        raise ValueError(f"Invalid timeframe: {request['timeframe']}. Must be one of {valid_timeframes}")
    
    # Validate dates
    start_date = request['start_date']
    end_date = request.get('end_date')
    
    if end_date and start_date > end_date:
        raise ValueError("start_date cannot be after end_date")
    
    # Validate intervention
    if not isinstance(request['intervention'], dict) or not request['intervention']:
        raise ValueError("intervention must be a non-empty dictionary")
    
    # Validate target_variables
    if not isinstance(request['target_variables'], list) or not request['target_variables']:
        raise ValueError("target_variables must be a non-empty list")
    
    # Validate target_variables are not in intervention
    for target in request['target_variables']:
        if target in request['intervention']:
            raise ValueError(f"Target variable '{target}' cannot be in the intervention")