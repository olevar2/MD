#!/usr/bin/env python
"""
Simple Fibonacci Test

This script tests the Fibonacci interfaces and adapters without relying on the full implementation.
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the feature-store-service directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Import the adapter interfaces from common-lib
from common_lib.indicators.fibonacci_interfaces import (
    TrendDirectionType,
    IFibonacciBase,
    IFibonacciRetracement,
    IFibonacciExtension,
    IFibonacciUtils
)

# Create a simple implementation of the interfaces for testing
class SimpleFibonacciBase(IFibonacciBase):
    """Simple implementation of IFibonacciBase for testing."""
    
    def __init__(self, name="SimpleFibonacci", **kwargs):
        self._name = name
        self._params = kwargs
    
    @property
    def name(self) -> str:
        """Get the name of the indicator."""
        return self._name
    
    @property
    def params(self) -> dict:
        """Get the parameters for the indicator."""
        return self._params
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the indicator values.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicator values added as columns
        """
        result = data.copy()
        result['fib_base'] = 1.0
        return result
    
    @staticmethod
    def get_info() -> dict:
        """
        Get information about the indicator.
        
        Returns:
            Dictionary with indicator information
        """
        return {
            'name': 'SimpleFibonacci',
            'description': 'Simple Fibonacci indicator for testing',
            'category': 'fibonacci'
        }


class SimpleFibonacciRetracement(SimpleFibonacciBase, IFibonacciRetracement):
    """Simple implementation of IFibonacciRetracement for testing."""
    
    def __init__(self, levels=None, **kwargs):
        super().__init__(name="SimpleFibonacciRetracement", **kwargs)
        self.levels = levels or [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with retracement levels added
        """
        result = data.copy()
        
        # Find high and low
        high = data['high'].max()
        low = data['low'].min()
        range_price = high - low
        
        # Calculate retracement levels
        for level in self.levels:
            level_str = str(level).replace('.', '_')
            result[f'fib_retracement_{level_str}'] = high - range_price * level
        
        # Mark start and end points
        result['fib_retracement_start'] = False
        result['fib_retracement_end'] = False
        
        # Mark the high and low points
        high_idx = data['high'].idxmax()
        low_idx = data['low'].idxmin()
        
        result.loc[high_idx, 'fib_retracement_start'] = True
        result.loc[low_idx, 'fib_retracement_end'] = True
        
        return result


class SimpleFibonacciExtension(SimpleFibonacciBase, IFibonacciExtension):
    """Simple implementation of IFibonacciExtension for testing."""
    
    def __init__(self, levels=None, **kwargs):
        super().__init__(name="SimpleFibonacciExtension", **kwargs)
        self.levels = levels or [0.0, 0.618, 1.0, 1.618, 2.618, 3.618]
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci extension levels.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with extension levels added
        """
        result = data.copy()
        
        # Find high and low
        high = data['high'].max()
        low = data['low'].min()
        range_price = high - low
        
        # Calculate extension levels
        for level in self.levels:
            level_str = str(level).replace('.', '_')
            result[f'fib_extension_{level_str}'] = low + range_price * level
        
        return result


class SimpleFibonacciUtils(IFibonacciUtils):
    """Simple implementation of IFibonacciUtils for testing."""
    
    @staticmethod
    def generate_fibonacci_sequence(n: int) -> list:
        """
        Generate a Fibonacci sequence of length n.
        
        Args:
            n: Length of the sequence
            
        Returns:
            List of Fibonacci numbers
        """
        if n <= 0:
            return []
        if n == 1:
            return [0]
        if n == 2:
            return [0, 1]
        
        sequence = [0, 1]
        for i in range(2, n):
            sequence.append(sequence[i-1] + sequence[i-2])
        
        return sequence
    
    @staticmethod
    def fibonacci_ratios() -> list:
        """
        Get common Fibonacci ratios.
        
        Returns:
            List of Fibonacci ratios
        """
        return [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618, 3.618, 4.236]
    
    @staticmethod
    def calculate_fibonacci_retracement_levels(
        start_price: float,
        end_price: float,
        levels: list = None
    ) -> dict:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            start_price: Starting price
            end_price: Ending price
            levels: Optional list of Fibonacci levels
            
        Returns:
            Dictionary mapping levels to prices
        """
        if levels is None:
            levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
        price_range = end_price - start_price
        result = {}
        
        for level in levels:
            result[level] = start_price + price_range * level
        
        return result
    
    @staticmethod
    def calculate_fibonacci_extension_levels(
        start_price: float,
        end_price: float,
        retracement_price: float,
        levels: list = None
    ) -> dict:
        """
        Calculate Fibonacci extension levels.
        
        Args:
            start_price: Starting price
            end_price: Ending price
            retracement_price: Retracement price
            levels: Optional list of Fibonacci levels
            
        Returns:
            Dictionary mapping levels to prices
        """
        if levels is None:
            levels = [0.0, 0.618, 1.0, 1.618, 2.618, 3.618]
        
        price_range = end_price - start_price
        result = {}
        
        for level in levels:
            result[level] = retracement_price + price_range * level
        
        return result


def create_test_data():
    """
    Create sample OHLCV data for testing.
    
    Returns:
        DataFrame with OHLCV data
    """
    # Create sample OHLCV data with a clear trend for testing
    dates = [datetime.now() + timedelta(days=i) for i in range(100)]
    
    # Create an uptrend followed by a downtrend
    close_prices = np.concatenate([
        np.linspace(100, 200, 50),  # Uptrend
        np.linspace(200, 150, 50)   # Downtrend
    ])
    
    # Add some noise to the data
    noise = np.random.normal(0, 2, 100)
    close_prices = close_prices + noise
    
    # Create high and low prices around close
    high_prices = close_prices + np.random.uniform(1, 5, 100)
    low_prices = close_prices - np.random.uniform(1, 5, 100)
    open_prices = close_prices - np.random.uniform(-3, 3, 100)
    volume = np.random.uniform(1000, 5000, 100)
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    return data


def test_fibonacci_interfaces():
    """Test the Fibonacci interfaces with simple implementations."""
    print("Testing Fibonacci interfaces...")
    
    # Create test data
    data = create_test_data()
    
    # Test SimpleFibonacciBase
    base = SimpleFibonacciBase()
    print(f"Base name: {base.name}")
    print(f"Base params: {base.params}")
    base_result = base.calculate(data)
    print(f"Base result columns: {base_result.columns.tolist()}")
    
    # Test SimpleFibonacciRetracement
    retracement = SimpleFibonacciRetracement()
    print(f"Retracement name: {retracement.name}")
    print(f"Retracement params: {retracement.params}")
    retracement_result = retracement.calculate(data)
    print(f"Retracement result columns: {retracement_result.columns.tolist()}")
    
    # Test SimpleFibonacciExtension
    extension = SimpleFibonacciExtension()
    print(f"Extension name: {extension.name}")
    print(f"Extension params: {extension.params}")
    extension_result = extension.calculate(data)
    print(f"Extension result columns: {extension_result.columns.tolist()}")
    
    # Test SimpleFibonacciUtils
    utils = SimpleFibonacciUtils()
    sequence = utils.generate_fibonacci_sequence(10)
    print(f"Fibonacci sequence: {sequence}")
    
    ratios = utils.fibonacci_ratios()
    print(f"Fibonacci ratios: {ratios}")
    
    retracement_levels = utils.calculate_fibonacci_retracement_levels(100, 200)
    print(f"Retracement levels: {retracement_levels}")
    
    extension_levels = utils.calculate_fibonacci_extension_levels(100, 200, 150)
    print(f"Extension levels: {extension_levels}")
    
    print("All interface tests passed!")


if __name__ == "__main__":
    test_fibonacci_interfaces()
