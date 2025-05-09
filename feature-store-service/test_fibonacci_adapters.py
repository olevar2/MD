#!/usr/bin/env python
"""
Test Fibonacci Adapters

This script tests the Fibonacci adapters to ensure they work correctly.
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
    IFibonacciRetracement,
    IFibonacciExtension,
    IFibonacciFan,
    IFibonacciTimeZones,
    IFibonacciCircles,
    IFibonacciClusters,
    IFibonacciUtils
)

# Import the adapter implementations from feature-store-service
from feature_store_service.adapters import (
    FibonacciRetracementAdapter,
    FibonacciExtensionAdapter,
    FibonacciFanAdapter,
    FibonacciTimeZonesAdapter,
    FibonacciCirclesAdapter,
    FibonacciClustersAdapter,
    FibonacciUtilsAdapter
)

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

def test_fibonacci_retracement_adapter():
    """Test FibonacciRetracementAdapter."""
    print("Testing FibonacciRetracementAdapter...")
    
    # Create test data
    data = create_test_data()
    
    # Create adapter
    adapter = FibonacciRetracementAdapter()
    
    # Test properties
    print(f"Name: {adapter.name}")
    print(f"Params: {adapter.params}")
    
    # Test calculate method
    result = adapter.calculate(data)
    
    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    
    # Check that result has more columns than input
    assert len(result.columns) > len(data.columns), "Result should have more columns than input"
    
    print("FibonacciRetracementAdapter test passed!")

def test_fibonacci_extension_adapter():
    """Test FibonacciExtensionAdapter."""
    print("Testing FibonacciExtensionAdapter...")
    
    # Create test data
    data = create_test_data()
    
    # Create adapter
    adapter = FibonacciExtensionAdapter()
    
    # Test properties
    print(f"Name: {adapter.name}")
    print(f"Params: {adapter.params}")
    
    # Test calculate method
    result = adapter.calculate(data)
    
    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    
    # Check that result has more columns than input
    assert len(result.columns) > len(data.columns), "Result should have more columns than input"
    
    print("FibonacciExtensionAdapter test passed!")

def test_fibonacci_utils_adapter():
    """Test FibonacciUtilsAdapter."""
    print("Testing FibonacciUtilsAdapter...")
    
    # Create adapter
    adapter = FibonacciUtilsAdapter()
    
    # Test generate_fibonacci_sequence
    sequence = adapter.generate_fibonacci_sequence(10)
    print(f"Fibonacci sequence: {sequence}")
    assert len(sequence) == 10, "Sequence should have 10 elements"
    
    # Test fibonacci_ratios
    ratios = adapter.fibonacci_ratios()
    print(f"Fibonacci ratios: {ratios}")
    assert len(ratios) > 0, "Ratios should not be empty"
    
    # Test calculate_fibonacci_retracement_levels
    levels = adapter.calculate_fibonacci_retracement_levels(100, 200)
    print(f"Retracement levels: {levels}")
    assert len(levels) > 0, "Levels should not be empty"
    
    # Test calculate_fibonacci_extension_levels
    levels = adapter.calculate_fibonacci_extension_levels(100, 200, 150)
    print(f"Extension levels: {levels}")
    assert len(levels) > 0, "Levels should not be empty"
    
    print("FibonacciUtilsAdapter test passed!")

if __name__ == "__main__":
    print("Testing Fibonacci adapters...")
    
    # Run tests
    test_fibonacci_retracement_adapter()
    test_fibonacci_extension_adapter()
    test_fibonacci_utils_adapter()
    
    print("All tests passed!")
