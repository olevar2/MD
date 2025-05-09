"""
Example script for using advanced pattern recognition.

This script demonstrates how to use the advanced pattern recognition capabilities
of the feature-store-service.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_store_service.indicators.advanced_patterns import (
    AdvancedPatternFacade,
    RenkoPatternRecognizer,
    IchimokuPatternRecognizer
)


def create_sample_data(days=200):
    """Create sample OHLCV data."""
    # Create dates
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    dates.reverse()
    
    # Create a simple uptrend followed by a downtrend
    close_prices = []
    for i in range(days):
        if i < days // 2:
            # Uptrend
            close_prices.append(100 + i * 0.5 + np.random.normal(0, 1))
        else:
            # Downtrend
            close_prices.append(100 + (days // 2) * 0.5 - (i - days // 2) * 0.3 + np.random.normal(0, 1))
    
    # Create high and low prices
    high_prices = [price + np.random.uniform(0.1, 0.5) for price in close_prices]
    low_prices = [price - np.random.uniform(0.1, 0.5) for price in close_prices]
    open_prices = [prev_close + np.random.normal(0, 0.2) for prev_close in [close_prices[0]] + close_prices[:-1]]
    volume = [1000 + np.random.randint(0, 500) for _ in range(days)]
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    return data


def plot_patterns(data, pattern_column, title):
    """Plot price data with pattern markers."""
    plt.figure(figsize=(12, 6))
    
    # Plot price data
    plt.plot(data.index, data['close'], label='Close Price')
    
    # Plot pattern markers
    pattern_indices = data[data[pattern_column] == 1].index
    if len(pattern_indices) > 0:
        plt.scatter(pattern_indices, data.loc[pattern_indices, 'close'], 
                   color='red', marker='^', s=100, label=pattern_column)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    return plt


def main():
    """Main function."""
    print("Creating sample data...")
    data = create_sample_data(days=200)
    
    print("Data shape:", data.shape)
    print("Data head:")
    print(data.head())
    
    # Example 1: Using the AdvancedPatternFacade
    print("\nExample 1: Using the AdvancedPatternFacade")
    facade = AdvancedPatternFacade(
        pattern_types=None,  # Use all patterns
        lookback_period=50,
        sensitivity=0.75
    )
    
    result = facade.calculate(data)
    
    # Count patterns
    pattern_cols = [col for col in result.columns if col.startswith('pattern_')]
    pattern_counts = {col: result[col].sum() for col in pattern_cols if result[col].sum() > 0}
    
    print("Detected patterns:")
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count} occurrences")
    
    # Plot a pattern if any were detected
    if pattern_counts:
        pattern_to_plot = max(pattern_counts.items(), key=lambda x: x[1])[0]
        plot = plot_patterns(result, pattern_to_plot, f"Price Chart with {pattern_to_plot} Pattern")
        plot.savefig("advanced_pattern_facade_example.png")
        print(f"Saved plot to advanced_pattern_facade_example.png")
    
    # Example 2: Using the RenkoPatternRecognizer
    print("\nExample 2: Using the RenkoPatternRecognizer")
    renko_recognizer = RenkoPatternRecognizer(
        brick_size=None,  # Auto-calculate
        brick_method='atr',
        atr_period=14,
        min_trend_length=3,
        min_consolidation_length=4,
        pattern_types=None,  # Use all patterns
        lookback_period=50,
        sensitivity=0.75
    )
    
    renko_result = renko_recognizer.calculate(data)
    
    # Count Renko patterns
    renko_pattern_cols = [col for col in renko_result.columns if col.startswith('pattern_renko_')]
    renko_pattern_counts = {col: renko_result[col].sum() for col in renko_pattern_cols if renko_result[col].sum() > 0}
    
    print("Detected Renko patterns:")
    for pattern, count in renko_pattern_counts.items():
        print(f"  {pattern}: {count} occurrences")
    
    # Plot a Renko pattern if any were detected
    if renko_pattern_counts:
        renko_pattern_to_plot = max(renko_pattern_counts.items(), key=lambda x: x[1])[0]
        renko_plot = plot_patterns(renko_result, renko_pattern_to_plot, f"Price Chart with {renko_pattern_to_plot} Pattern")
        renko_plot.savefig("renko_pattern_example.png")
        print(f"Saved plot to renko_pattern_example.png")
    
    # Example 3: Using the IchimokuPatternRecognizer
    print("\nExample 3: Using the IchimokuPatternRecognizer")
    ichimoku_recognizer = IchimokuPatternRecognizer(
        tenkan_period=9,
        kijun_period=26,
        senkou_b_period=52,
        displacement=26,
        pattern_types=None,  # Use all patterns
        lookback_period=100,
        sensitivity=0.75
    )
    
    ichimoku_result = ichimoku_recognizer.calculate(data)
    
    # Count Ichimoku patterns
    ichimoku_pattern_cols = [col for col in ichimoku_result.columns if col.startswith('pattern_ichimoku_')]
    ichimoku_pattern_counts = {col: ichimoku_result[col].sum() for col in ichimoku_pattern_cols if ichimoku_result[col].sum() > 0}
    
    print("Detected Ichimoku patterns:")
    for pattern, count in ichimoku_pattern_counts.items():
        print(f"  {pattern}: {count} occurrences")
    
    # Plot an Ichimoku pattern if any were detected
    if ichimoku_pattern_counts:
        ichimoku_pattern_to_plot = max(ichimoku_pattern_counts.items(), key=lambda x: x[1])[0]
        ichimoku_plot = plot_patterns(ichimoku_result, ichimoku_pattern_to_plot, f"Price Chart with {ichimoku_pattern_to_plot} Pattern")
        ichimoku_plot.savefig("ichimoku_pattern_example.png")
        print(f"Saved plot to ichimoku_pattern_example.png")
    
    print("\nExample complete!")


if __name__ == "__main__":
    main()