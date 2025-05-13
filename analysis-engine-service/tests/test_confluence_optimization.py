"""
Simple script to test the performance of the optimized confluence analyzer.
"""
import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
    '..')))
try:
    from analysis_engine.analysis.confluence.confluence_analyzer import ConfluenceAnalyzer
    print('Successfully imported ConfluenceAnalyzer')
except ImportError as e:
    print(f'Error importing ConfluenceAnalyzer: {e}')
    print('Trying alternative import path...')
    try:
        sys.path.insert(0, 'D:\\MD\\forex_trading_platform')
        from analysis_engine.analysis.confluence.confluence_analyzer import ConfluenceAnalyzer
        print('Successfully imported ConfluenceAnalyzer using full path')
    except ImportError as e:
        print(f'Error importing ConfluenceAnalyzer with full path: {e}')
        sys.exit(1)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

def generate_sample_data(num_bars=500):
    """Generate sample market data for testing."""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=num_bars)
    timestamps = [(start_time + timedelta(hours=i)).isoformat() for i in
        range(num_bars)]
    np.random.seed(42)
    base_price = 1.2
    random_walk = np.random.normal(0, 0.0002, num_bars).cumsum()
    trend = np.linspace(0, 0.01, num_bars)
    cycles = 0.005 * np.sin(np.linspace(0, 5 * np.pi, num_bars))
    close_prices = base_price + random_walk + trend + cycles
    high_prices = close_prices + np.random.uniform(0, 0.0015, num_bars)
    low_prices = close_prices - np.random.uniform(0, 0.0015, num_bars)
    open_prices = low_prices + np.random.uniform(0, 0.0015, num_bars)
    volume = np.random.uniform(100, 1000, num_bars)
    market_data = {'timestamp': timestamps, 'open': open_prices.tolist(),
        'high': high_prices.tolist(), 'low': low_prices.tolist(), 'close':
        close_prices.tolist(), 'volume': volume.tolist()}
    df = pd.DataFrame(market_data)
    return df


@with_exception_handling
def test_confluence_analyzer_performance():
    """Test the performance of the ConfluenceAnalyzer."""
    print('Creating ConfluenceAnalyzer instance...')
    analyzer = ConfluenceAnalyzer()
    print('Generating sample data...')
    df = generate_sample_data(num_bars=500)
    print('Running performance test...')
    num_runs = 5
    total_time = 0
    for i in range(num_runs):
        print(f'Run {i + 1}/{num_runs}...')
        start_time = time.time()
        try:
            result = analyzer.analyze(df)
            if result.is_valid:
                print('Analysis successful')
                if 'performance_metrics' in result.result:
                    metrics = result.result['performance_metrics']
                    print('Performance metrics:')
                    for key, value in metrics.items():
                        print(f'  {key}: {value:.4f} seconds')
            else:
                print(
                    f"Analysis failed: {result.result.get('error', 'Unknown error')}"
                    )
        except Exception as e:
            print(f'Error during analysis: {e}')
            continue
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'Execution time: {execution_time:.4f} seconds')
        total_time += execution_time
    if num_runs > 0:
        avg_time = total_time / num_runs
        print(f'\nAverage execution time: {avg_time:.4f} seconds')
    print('Performance test completed')


if __name__ == '__main__':
    print('Starting confluence analyzer performance test...')
    test_confluence_analyzer_performance()
