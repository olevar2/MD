"""
Fibonacci Time Zones module.

This module provides implementation of Fibonacci time zones analysis.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from core.base_4 import FibonacciBase, find_swing_highs, find_swing_lows


class FibonacciTimeZones(FibonacciBase):
    """
    Fibonacci Time Zones
    
    Projects potential future significant time points based on Fibonacci numbers.
    This tool helps identify time periods where significant market reversals 
    or continuations might occur.
    """
    
    def __init__(
        self, 
        fib_sequence: Optional[List[int]] = None,
        starting_point: Optional[int] = None,
        auto_detect_start: bool = True,
        max_zones: int = 8,
        **kwargs
    ):
        """
        Initialize Fibonacci Time Zones indicator.
        
        Args:
            fib_sequence: List of Fibonacci numbers to use for time projections
            starting_point: Manual index for the starting point (None for auto-detect)
            auto_detect_start: Whether to automatically detect the starting point
            max_zones: Maximum number of time zones to project
            **kwargs: Additional parameters
        """
        name = kwargs.pop('name', 'fib_time_zone')
        super().__init__(name=name, **kwargs)
        
        # Define default Fibonacci sequence numbers
        self.fib_sequence = fib_sequence or [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        self.starting_point = starting_point
        self.auto_detect_start = auto_detect_start
        self.max_zones = max(1, min(max_zones, 10))  # Limit between 1 and 10
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci Time Zones for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Fibonacci Time Zone values
        """
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Determine the starting point
        if self.starting_point is not None:
            # Use manually specified starting point
            start_pos = min(self.starting_point, len(result) - 1)
        elif self.auto_detect_start:
            # Automatically detect a significant starting point
            # For simplicity, use a major swing point
            swing_highs = find_swing_highs(result, n=30)
            swing_lows = find_swing_lows(result, n=30)
            
            # Combine and sort by significance (price movement)
            all_swings = []
            for idx, price in swing_highs:
                all_swings.append((idx, price, 'high'))
            for idx, price in swing_lows:
                all_swings.append((idx, price, 'low'))
            
            # Sort by index to get chronological order
            all_swings.sort(key=lambda x: x[0])
            
            if all_swings:
                # Use the first significant swing point
                start_pos = all_swings[0][0]
                if isinstance(start_pos, pd.Timestamp):
                    # If it's already a timestamp, use it directly
                    start_idx = start_pos
                else:
                    # Convert to integer index if needed
                    start_pos = int(start_pos)
                    start_idx = data.index[start_pos]
            else:
                # Default to the first point
                start_pos = 0
                start_idx = data.index[start_pos]
        else:
            # Default to the first point
            start_pos = 0
            start_idx = data.index[start_pos]
        
        # Create column for Fibonacci time zones
        result['fib_time_zone'] = 0
        
        # Add time zones based on Fibonacci numbers
        for i, fib in enumerate(self.fib_sequence[:self.max_zones]):
            zone_pos = int(start_pos) + fib
            if zone_pos < len(result):
                zone_idx = result.index[zone_pos]
                result.loc[zone_idx, 'fib_time_zone'] = i + 1  # Zone number
                
        # Mark the starting point
        result['fib_time_zone_start'] = False
        result.loc[start_idx, 'fib_time_zone_start'] = True
        
        # Add information about each zone
        for i, fib in enumerate(self.fib_sequence[:self.max_zones]):
            col_name = f"fib_time_zone_{i+1}"
            result[col_name] = False
            
            zone_pos = int(start_pos) + fib
            if zone_pos < len(result):
                zone_idx = result.index[zone_pos]
                result.loc[zone_idx, col_name] = True
        
        return result
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Fibonacci Time Zones',
            'description': 'Projects potential future significant time points based on Fibonacci numbers',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'fib_sequence',
                    'description': 'List of Fibonacci numbers to use for time projections',
                    'type': 'list',
                    'default': [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
                },
                {
                    'name': 'starting_point',
                    'description': 'Manual index for the starting point',
                    'type': 'int',
                    'default': None
                },
                {
                    'name': 'auto_detect_start',
                    'description': 'Whether to automatically detect the starting point',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'max_zones',
                    'description': 'Maximum number of time zones to project',
                    'type': 'int',
                    'default': 8
                }
            ]
        }