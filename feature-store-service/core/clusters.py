"""
Fibonacci Clusters module.

This module provides implementation of Fibonacci clusters analysis.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from core.base_4 import FibonacciBase


class FibonacciClusters(FibonacciBase):
    """
    Fibonacci Clusters

    Identifies price zones where multiple Fibonacci levels (from different tools
    like retracements, extensions, projections, fans, etc.) converge.
    These cluster zones are considered stronger potential support/resistance areas.
    Requires results from other Fibonacci indicators as input columns in the DataFrame.
    """

    def __init__(
        self,
        price_column: str = 'close',
        cluster_threshold: int = 3,
        price_tolerance: float = 0.5,  # Percentage tolerance for price clustering
        **kwargs
    ):
        """
        Initialize Fibonacci Clusters indicator.
        
        Args:
            price_column: Column name for price data
            cluster_threshold: Minimum number of Fibonacci levels to form a cluster
            price_tolerance: Percentage tolerance for price clustering
            **kwargs: Additional parameters
        """
        name = kwargs.pop('name', 'fib_cluster')
        super().__init__(name=name, **kwargs)
        
        self.price_column = price_column
        self.cluster_threshold = max(2, cluster_threshold)  # Minimum 2 levels for a cluster
        self.price_tolerance = max(0.1, min(price_tolerance, 5.0)) / 100.0  # Convert to decimal, limit range
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci Clusters for the given data.
        
        Args:
            data: DataFrame with OHLCV data and Fibonacci level columns
            
        Returns:
            DataFrame with Fibonacci Cluster values
        """
        if self.price_column not in data.columns:
            raise ValueError(f"Data must contain '{self.price_column}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Find all Fibonacci level columns
        fib_columns = [col for col in result.columns if 
                      ('fib_retracement_' in col or 
                       'fib_extension_' in col or 
                       'fib_fan_' in col or 
                       'fib_circle_' in col) and 
                      not col.endswith('_start') and 
                      not col.endswith('_end') and 
                      not col.endswith('_trend') and 
                      not col.endswith('_retracement')]
        
        if len(fib_columns) < self.cluster_threshold:
            # Not enough Fibonacci levels to form clusters
            result[f'{self.name}_strength'] = 0
            result[f'{self.name}_count'] = 0
            return result
        
        # Initialize cluster columns
        result[f'{self.name}_strength'] = 0.0
        result[f'{self.name}_count'] = 0
        
        # Calculate average price for the dataset
        avg_price = result[self.price_column].mean()
        
        # Calculate tolerance in absolute price
        tolerance = avg_price * self.price_tolerance
        
        # For each row, find clusters of Fibonacci levels
        for i in range(len(result)):
            # Get current price
            current_price = result.iloc[i][self.price_column]
            
            # Count Fibonacci levels near the current price
            level_count = 0
            level_prices = []
            
            for col in fib_columns:
                level_price = result.iloc[i][col]
                if pd.notna(level_price) and abs(level_price - current_price) <= tolerance:
                    level_count += 1
                    level_prices.append(level_price)
            
            # If enough levels form a cluster
            if level_count >= self.cluster_threshold:
                # Calculate cluster strength (inverse of average distance)
                if level_prices:
                    avg_distance = np.mean([abs(p - current_price) for p in level_prices])
                    if avg_distance > 0:
                        strength = 1.0 / (avg_distance / current_price * 100.0)
                    else:
                        strength = 10.0  # Maximum strength for exact matches
                else:
                    strength = 0.0
                
                # Update cluster information
                result.iloc[i][f'{self.name}_count'] = level_count
                result.iloc[i][f'{self.name}_strength'] = strength
        
        return result
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Fibonacci Clusters',
            'description': 'Identifies price zones where multiple Fibonacci levels converge',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'price_column',
                    'description': 'Column name for price data',
                    'type': 'str',
                    'default': 'close'
                },
                {
                    'name': 'cluster_threshold',
                    'description': 'Minimum number of Fibonacci levels to form a cluster',
                    'type': 'int',
                    'default': 3
                },
                {
                    'name': 'price_tolerance',
                    'description': 'Percentage tolerance for price clustering',
                    'type': 'float',
                    'default': 0.5
                }
            ]
        }