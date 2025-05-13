"""
Profile the performance of the ConfluenceAnalyzer.

This script profiles the performance of the ConfluenceAnalyzer class
and generates a performance report.
"""

import asyncio
import cProfile
import pstats
import io
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis_engine.analysis.confluence.confluence_analyzer import ConfluenceAnalyzer


def generate_sample_data(num_bars=500):
    """Generate sample market data for testing."""
    # Generate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=num_bars)
    timestamps = [
        (start_time + timedelta(hours=i)).isoformat()
        for i in range(num_bars)
    ]
    
    # Generate price data with some realistic patterns
    np.random.seed(42)  # For reproducibility
    
    # Start with a base price
    base_price = 1.2000
    
    # Generate random walk
    random_walk = np.random.normal(0, 0.0002, num_bars).cumsum()
    
    # Add trend
    trend = np.linspace(0, 0.01, num_bars)
    
    # Add some cyclical patterns
    cycles = 0.005 * np.sin(np.linspace(0, 5 * np.pi, num_bars))
    
    # Combine components
    close_prices = base_price + random_walk + trend + cycles
    
    # Generate OHLC data
    high_prices = close_prices + np.random.uniform(0, 0.0015, num_bars)
    low_prices = close_prices - np.random.uniform(0, 0.0015, num_bars)
    open_prices = low_prices + np.random.uniform(0, 0.0015, num_bars)
    
    # Generate volume data
    volume = np.random.uniform(100, 1000, num_bars)
    
    # Create market data dictionary
    market_data = {
        "timestamp": timestamps,
        "open": open_prices.tolist(),
        "high": high_prices.tolist(),
        "low": low_prices.tolist(),
        "close": close_prices.tolist(),
        "volume": volume.tolist()
    }
    
    # Create full data dictionary
    data = {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "market_data": market_data,
        "market_regime": "trending"
    }
    
    return data


async def run_analysis(analyzer, data):
    """Run the analysis and return the result."""
    return await analyzer.analyze(data)


def profile_analyzer(data_sizes=[100, 500, 1000]):
    """Profile the analyzer with different data sizes."""
    analyzer = ConfluenceAnalyzer()
    
    results = {}
    
    for size in data_sizes:
        print(f"\nProfiling with {size} bars of data...")
        data = generate_sample_data(num_bars=size)
        
        # Run once to warm up cache
        asyncio.run(run_analysis(analyzer, data))
        
        # Profile the analysis
        pr = cProfile.Profile()
        pr.enable()
        
        start_time = time.time()
        result = asyncio.run(run_analysis(analyzer, data))
        execution_time = time.time() - start_time
        
        pr.disable()
        
        # Get profiling stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Print top 20 functions by cumulative time
        
        # Store results
        results[size] = {
            'execution_time': execution_time,
            'profile_stats': s.getvalue(),
            'performance_metrics': result.result.get('performance_metrics', {}),
            'zone_count': len(result.result.get('confluence_zones', []))
        }
        
        print(f"Analysis completed in {execution_time:.4f} seconds")
        print(f"Found {results[size]['zone_count']} confluence zones")
        
    return results


def plot_performance_results(results):
    """Plot performance results."""
    sizes = list(results.keys())
    times = [results[size]['execution_time'] for size in sizes]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'o-', linewidth=2)
    plt.xlabel('Number of Bars')
    plt.ylabel('Execution Time (seconds)')
    plt.title('ConfluenceAnalyzer Performance')
    plt.grid(True)
    
    # Add data labels
    for i, size in enumerate(sizes):
        plt.annotate(f"{times[i]:.3f}s", 
                    (size, times[i]), 
                    textcoords="offset points",
                    xytext=(0, 10), 
                    ha='center')
    
    # Save the plot
    plt.savefig('confluence_analyzer_performance.png')
    print("\nPerformance plot saved to 'confluence_analyzer_performance.png'")


def main():
    """Main function to run the profiling."""
    print("Profiling ConfluenceAnalyzer performance...")
    
    # Profile with different data sizes
    results = profile_analyzer(data_sizes=[100, 500, 1000, 2000])
    
    # Plot results
    plot_performance_results(results)
    
    # Print detailed profiling information
    print("\nDetailed profiling information:")
    for size, result in results.items():
        print(f"\n{'='*50}")
        print(f"Profile for {size} bars:")
        print(f"{'='*50}")
        print(result['profile_stats'])
        
        print("\nPerformance metrics:")
        for key, value in result['performance_metrics'].items():
            print(f"  {key}: {value:.4f} seconds")


if __name__ == "__main__":
    main()
