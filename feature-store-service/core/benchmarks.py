"""
Performance Benchmarking for Gann Tools.

This module provides benchmarking utilities to compare the performance of the original
and refactored Gann tools implementations.
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Callable

# Import both original and refactored implementations
import utils.gann_tools as original
from feature_store_service.indicators.gann import (
    GannAngles as NewGannAngles,
    GannFan as NewGannFan,
    GannSquare as NewGannSquare,
    GannTimeCycles as NewGannTimeCycles,
    GannGrid as NewGannGrid,
    GannBox as NewGannBox,
    GannSquare144 as NewGannSquare144,
    GannHexagon as NewGannHexagon
)


def generate_test_data(num_bars: int = 1000) -> pd.DataFrame:
    """
    Generate test data for benchmarking.

    Args:
        num_bars: Number of bars to generate

    Returns:
        DataFrame with OHLCV data
    """
    dates = [datetime.now() - timedelta(days=i) for i in range(num_bars)]
    dates.reverse()  # Oldest first

    # Create a simple uptrend with some volatility
    base_price = 100.0
    trend = np.linspace(0, 50, num_bars)  # Uptrend from 100 to 150
    volatility = np.random.normal(0, 2, num_bars)  # Add some noise

    prices = base_price + trend + volatility

    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices - np.random.uniform(0, 1, num_bars),
        'high': prices + np.random.uniform(1, 2, num_bars),
        'low': prices - np.random.uniform(1, 2, num_bars),
        'close': prices + np.random.uniform(0, 1, num_bars),
        'volume': np.random.uniform(1000, 5000, num_bars)
    }, index=dates)

    return data


def benchmark_function(func: Callable, *args, **kwargs) -> Tuple[float, Any]:
    """
    Benchmark a function's execution time.

    Args:
        func: Function to benchmark
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Tuple of (execution_time, function_result)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    execution_time = end_time - start_time

    return execution_time, result


def benchmark_gann_angles(data: pd.DataFrame, num_runs: int = 5) -> Dict[str, float]:
    """
    Benchmark GannAngles implementations.

    Args:
        data: DataFrame with OHLCV data
        num_runs: Number of benchmark runs

    Returns:
        Dictionary with benchmark results
    """
    # Initialize implementations
    orig_gann_angles = original.GannAngles(
        pivot_type="swing_low",
        angle_types=["1x1", "1x2", "2x1", "1x4", "4x1", "1x8", "8x1"],
        lookback_period=100,
        price_scaling=1.0,
        projection_bars=50
    )

    new_gann_angles = NewGannAngles(
        pivot_type="swing_low",
        angle_types=["1x1", "1x2", "2x1", "1x4", "4x1", "1x8", "8x1"],
        lookback_period=100,
        price_scaling=1.0,
        projection_bars=50
    )

    # Run benchmarks
    orig_times = []
    new_times = []

    for _ in range(num_runs):
        # Original implementation uses calculate_angles
        orig_time, _ = benchmark_function(orig_gann_angles.calculate_angles, data)
        # New implementation uses calculate
        new_time, _ = benchmark_function(new_gann_angles.calculate, data)

        orig_times.append(orig_time)
        new_times.append(new_time)

    # Calculate average times
    avg_orig_time = sum(orig_times) / num_runs
    avg_new_time = sum(new_times) / num_runs

    return {
        'original': avg_orig_time,
        'refactored': avg_new_time,
        'improvement': (avg_orig_time - avg_new_time) / avg_orig_time * 100
    }


def benchmark_gann_fan(data: pd.DataFrame, num_runs: int = 5) -> Dict[str, float]:
    """
    Benchmark GannFan implementations.

    Args:
        data: DataFrame with OHLCV data
        num_runs: Number of benchmark runs

    Returns:
        Dictionary with benchmark results
    """
    # Initialize implementations
    orig_gann_fan = original.GannFan(
        pivot_type="swing_low",
        fan_angles=["1x1", "1x2", "2x1", "1x4", "4x1", "1x8", "8x1"],
        lookback_period=100,
        price_scaling=1.0,
        projection_bars=50
    )

    new_gann_fan = NewGannFan(
        pivot_type="swing_low",
        fan_angles=["1x1", "1x2", "2x1", "1x4", "4x1", "1x8", "8x1"],
        lookback_period=100,
        price_scaling=1.0,
        projection_bars=50
    )

    # Run benchmarks
    orig_times = []
    new_times = []

    for _ in range(num_runs):
        # Original implementation uses calculate_fan
        orig_time, _ = benchmark_function(orig_gann_fan.calculate_fan, data)
        # New implementation uses calculate
        new_time, _ = benchmark_function(new_gann_fan.calculate, data)

        orig_times.append(orig_time)
        new_times.append(new_time)

    # Calculate average times
    avg_orig_time = sum(orig_times) / num_runs
    avg_new_time = sum(new_times) / num_runs

    return {
        'original': avg_orig_time,
        'refactored': avg_new_time,
        'improvement': (avg_orig_time - avg_new_time) / avg_orig_time * 100
    }


def benchmark_gann_square(data: pd.DataFrame, num_runs: int = 5) -> Dict[str, float]:
    """
    Benchmark GannSquare implementations.

    Args:
        data: DataFrame with OHLCV data
        num_runs: Number of benchmark runs

    Returns:
        Dictionary with benchmark results
    """
    # Initialize implementations
    orig_gann_square = original.GannSquare9(
        base_price=100.0  # Use a fixed base price for benchmarking
    )

    new_gann_square = NewGannSquare(
        square_type="square_of_9",
        pivot_price=100.0,  # Use the same base price
        auto_detect_pivot=False,
        lookback_period=100,
        num_levels=4
    )

    # Run benchmarks
    orig_times = []
    new_times = []

    for _ in range(num_runs):
        # Original implementation uses different methods
        orig_time, _ = benchmark_function(orig_gann_square.calculate_levels, n_levels=4)
        # New implementation uses calculate
        new_time, _ = benchmark_function(new_gann_square.calculate, data)

        orig_times.append(orig_time)
        new_times.append(new_time)

    # Calculate average times
    avg_orig_time = sum(orig_times) / num_runs
    avg_new_time = sum(new_times) / num_runs

    return {
        'original': avg_orig_time,
        'refactored': avg_new_time,
        'improvement': (avg_orig_time - avg_new_time) / avg_orig_time * 100
    }


def run_all_benchmarks(data_sizes: List[int] = [100, 500, 1000, 5000]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Run all benchmarks for different data sizes.

    Args:
        data_sizes: List of data sizes to benchmark

    Returns:
        Dictionary with benchmark results
    """
    results = {}

    for size in data_sizes:
        print(f"Running benchmarks for data size: {size}")
        data = generate_test_data(size)

        results[size] = {
            'GannAngles': benchmark_gann_angles(data),
            'GannFan': benchmark_gann_fan(data),
            'GannSquare': benchmark_gann_square(data)
        }

        # Print results
        print(f"  GannAngles: Original={results[size]['GannAngles']['original']:.4f}s, Refactored={results[size]['GannAngles']['refactored']:.4f}s, Improvement={results[size]['GannAngles']['improvement']:.2f}%")
        print(f"  GannFan: Original={results[size]['GannFan']['original']:.4f}s, Refactored={results[size]['GannFan']['refactored']:.4f}s, Improvement={results[size]['GannFan']['improvement']:.2f}%")
        print(f"  GannSquare: Original={results[size]['GannSquare']['original']:.4f}s, Refactored={results[size]['GannSquare']['refactored']:.4f}s, Improvement={results[size]['GannSquare']['improvement']:.2f}%")

    return results


def plot_benchmark_results(results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """
    Plot benchmark results.

    Args:
        results: Dictionary with benchmark results
    """
    data_sizes = list(results.keys())
    tools = list(results[data_sizes[0]].keys())

    fig, axes = plt.subplots(len(tools), 1, figsize=(10, 4 * len(tools)))

    for i, tool in enumerate(tools):
        ax = axes[i] if len(tools) > 1 else axes

        orig_times = [results[size][tool]['original'] for size in data_sizes]
        new_times = [results[size][tool]['refactored'] for size in data_sizes]

        ax.plot(data_sizes, orig_times, 'o-', label='Original')
        ax.plot(data_sizes, new_times, 'o-', label='Refactored')

        ax.set_title(f'{tool} Performance')
        ax.set_xlabel('Data Size (bars)')
        ax.set_ylabel('Execution Time (seconds)')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('gann_benchmarks.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    # Run benchmarks
    results = run_all_benchmarks()

    # Plot results
    plot_benchmark_results(results)
