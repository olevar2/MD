"""
Real-world performance test for optimized components.

This script tests the performance of optimized components with real-world data.
It downloads historical forex data and runs performance tests on the optimized
confluence detector and related components.

Usage:
    python real_world_performance_test.py [--pairs=8] [--days=30] [--iterations=5]
"""
import asyncio
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import argparse
import tracemalloc
import logging
import requests
import zipfile
import io
from typing import Dict, List, Any, Optional, Tuple, Union
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
try:
    from analysis_engine.multi_asset.related_pairs_confluence_detector import RelatedPairsConfluenceAnalyzer
    from analysis_engine.multi_asset.optimized_confluence_detector import OptimizedConfluenceDetector
    from analysis_engine.multi_asset.currency_strength_analyzer import CurrencyStrengthAnalyzer
    from analysis_engine.utils.memory_optimized_dataframe import MemoryOptimizedDataFrame
    from analysis_engine.utils.distributed_tracing import DistributedTracer
    from analysis_engine.utils.gpu_accelerator import GPUAccelerator
    logger.info('Successfully imported modules')
except ImportError as e:
    logger.error(f'Error importing modules: {e}')
    logger.info('Trying alternative import path...')
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(
            __file__), '../..'))
        sys.path.insert(0, project_root)
        from analysis_engine.multi_asset.related_pairs_confluence_detector import RelatedPairsConfluenceAnalyzer
        from analysis_engine.multi_asset.optimized_confluence_detector import OptimizedConfluenceDetector
        from analysis_engine.multi_asset.currency_strength_analyzer import CurrencyStrengthAnalyzer
        from analysis_engine.utils.memory_optimized_dataframe import MemoryOptimizedDataFrame
        from analysis_engine.utils.distributed_tracing import DistributedTracer
        from analysis_engine.utils.gpu_accelerator import GPUAccelerator
        logger.info(
            f'Successfully imported modules using project root: {project_root}'
            )
    except ImportError as e:
        logger.error(f'Error importing modules with project root path: {e}')
        sys.exit(1)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class MockCorrelationService:
    """Mock correlation service for testing."""

    @with_resilience('get_all_correlations')
    async def get_all_correlations(self):
        """Return mock correlation data."""
        pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURGBP',
            'EURJPY', 'GBPJPY']
        correlations = {}
        for pair1 in pairs:
            correlations[pair1] = {}
            for pair2 in pairs:
                if pair1 != pair2:
                    if pair1[:3] == pair2[:3] or pair1[3:] == pair2[3:]:
                        correlations[pair1][pair2] = np.random.uniform(0.6, 0.9
                            )
                    elif pair1[:3] in pair2[3:] or pair1[3:] in pair2[:3]:
                        correlations[pair1][pair2] = np.random.uniform(-0.9,
                            -0.6)
                    else:
                        correlations[pair1][pair2] = np.random.uniform(-0.5,
                            0.5)
        return correlations


@with_exception_handling
def download_historical_data(pairs, days=30):
    """
    Download historical forex data for testing.
    
    Args:
        pairs: List of currency pairs
        days: Number of days of historical data
        
    Returns:
        Dictionary mapping currency pairs to price DataFrames
    """
    logger.info(
        f'Downloading historical data for {len(pairs)} pairs, {days} days')
    price_data = {}
    api_key = 'demo'
    for pair in pairs:
        try:
            formatted_pair = f'{pair[:3]}/{pair[3:6]}'
            logger.info(f'Downloading data for {formatted_pair}')
            url = (
                f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={pair[:3]}&to_symbol={pair[3:6]}&outputsize=full&apikey={api_key}&datatype=csv'
                )
            response = requests.get(url)
            if response.status_code != 200:
                logger.warning(
                    f'Failed to download data for {formatted_pair}: {response.status_code}'
                    )
                continue
            df = pd.read_csv(io.StringIO(response.text))
            df = df.rename(columns={'timestamp': 'timestamp', 'open':
                'open', 'high': 'high', 'low': 'low', 'close': 'close'})
            df['volume'] = np.random.uniform(100, 1000, len(df))
            df = df.head(days)
            price_data[pair] = df
            logger.info(f'Downloaded {len(df)} rows for {formatted_pair}')
            time.sleep(15)
        except Exception as e:
            logger.error(f'Error downloading data for {pair}: {e}')
    if not price_data:
        logger.warning('No data downloaded, generating synthetic data')
        price_data = generate_synthetic_data(pairs, days * 24)
    return price_data


def generate_synthetic_data(pairs, bars_per_pair=720):
    """
    Generate synthetic price data for testing when real data is not available.
    
    Args:
        pairs: List of currency pairs
        bars_per_pair: Number of price bars per pair
        
    Returns:
        Dictionary mapping currency pairs to price DataFrames
    """
    logger.info(
        f'Generating synthetic data for {len(pairs)} pairs, {bars_per_pair} bars per pair'
        )
    price_data = {}
    for pair in pairs:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=bars_per_pair)
        timestamps = [(start_time + timedelta(hours=i)).isoformat() for i in
            range(bars_per_pair)]
        np.random.seed(42 + ord(pair[0]))
        if pair.endswith('JPY'):
            base_price = 110.0
        else:
            base_price = 1.2
        random_walk = np.random.normal(0, 0.0002, bars_per_pair).cumsum()
        trend = np.linspace(0, 0.01, bars_per_pair)
        cycles = 0.005 * np.sin(np.linspace(0, 5 * np.pi, bars_per_pair))
        close_prices = base_price + random_walk + trend + cycles
        high_prices = close_prices + np.random.uniform(0, 0.0015, bars_per_pair
            )
        low_prices = close_prices - np.random.uniform(0, 0.0015, bars_per_pair)
        open_prices = low_prices + np.random.uniform(0, 0.0015, bars_per_pair)
        volume = np.random.uniform(100, 1000, bars_per_pair)
        df = pd.DataFrame({'timestamp': timestamps, 'open': open_prices,
            'high': high_prices, 'low': low_prices, 'close': close_prices,
            'volume': volume})
        price_data[pair] = df
    return price_data


async def run_performance_test(pairs=8, days=30, iterations=5, use_gpu=False):
    """
    Run performance tests with real-world data.
    
    Args:
        pairs: Number of currency pairs to include
        days: Number of days of historical data
        iterations: Number of iterations for each test
        use_gpu: Whether to use GPU acceleration
    """
    logger.info(
        f'Starting performance test with {pairs} pairs, {days} days, {iterations} iterations, use_gpu={use_gpu}'
        )
    all_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURGBP',
        'EURJPY', 'GBPJPY']
    test_pairs = all_pairs[:pairs]
    price_data = download_historical_data(test_pairs, days)
    correlation_service = MockCorrelationService()
    currency_strength_analyzer = CurrencyStrengthAnalyzer()
    gpu_accelerator = None
    if use_gpu:
        gpu_accelerator = GPUAccelerator(enable_gpu=True, memory_limit_mb=
            1024, batch_size=1000)
    tracer = DistributedTracer(service_name='performance-test',
        enable_tracing=True, sampling_rate=0.1)
    original_analyzer = RelatedPairsConfluenceAnalyzer(correlation_service=
        correlation_service, currency_strength_analyzer=
        currency_strength_analyzer, correlation_threshold=0.7,
        lookback_periods=20, cache_ttl_minutes=60, max_workers=4)
    optimized_analyzer = OptimizedConfluenceDetector(correlation_service=
        correlation_service, currency_strength_analyzer=
        currency_strength_analyzer, correlation_threshold=0.7,
        lookback_periods=20, cache_ttl_minutes=60, max_workers=4)
    related_pairs = await original_analyzer.find_related_pairs('EURUSD')
    optimized_price_data = {pair: MemoryOptimizedDataFrame(df).
        optimize_dtypes() for pair, df in price_data.items()}
    results = {'confluence': {'original': {'cold': [], 'warm': []},
        'optimized': {'cold': [], 'warm': []}}, 'divergence': {'original':
        {'cold': [], 'warm': []}, 'optimized': {'cold': [], 'warm': []}},
        'memory': {'original': {'confluence': 0, 'divergence': 0},
        'optimized': {'confluence': 0, 'divergence': 0}}}
    signal_types = ['trend', 'reversal', 'breakout']
    signal_directions = ['bullish', 'bearish']
    logger.info('\nBenchmarking confluence detection...')
    for signal_type in signal_types:
        for signal_direction in signal_directions:
            logger.info(f'\n  Testing {signal_type} {signal_direction}...')
            logger.info('    Original implementation (cold cache)...')
            original_analyzer.related_pairs_cache.clear()
            tracemalloc.start()
            start_time = time.time()
            for i in range(iterations):
                with tracer.start_span(
                    f'original_confluence_{signal_type}_{signal_direction}_cold'
                    ):
                    result = original_analyzer.detect_confluence(symbol=
                        'EURUSD', price_data=price_data, signal_type=
                        signal_type, signal_direction=signal_direction,
                        related_pairs=related_pairs)
            cold_time = (time.time() - start_time) / iterations
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            results['confluence']['original']['cold'].append(cold_time)
            results['memory']['original']['confluence'] = peak / 1024 / 1024
            logger.info(
                f'      Time: {cold_time:.4f}s, Memory: {peak / 1024 / 1024:.2f}MB'
                )
            logger.info(
                f"      Confluence score: {result.get('confluence_score', 0):.4f}"
                )
            logger.info(
                f"      Confirmations: {result.get('confirmation_count', 0)}, Contradictions: {result.get('contradiction_count', 0)}"
                )
            logger.info('    Original implementation (warm cache)...')
            start_time = time.time()
            for i in range(iterations):
                with tracer.start_span(
                    f'original_confluence_{signal_type}_{signal_direction}_warm'
                    ):
                    result = original_analyzer.detect_confluence(symbol=
                        'EURUSD', price_data=price_data, signal_type=
                        signal_type, signal_direction=signal_direction,
                        related_pairs=related_pairs)
            warm_time = (time.time() - start_time) / iterations
            results['confluence']['original']['warm'].append(warm_time)
            logger.info(f'      Time: {warm_time:.4f}s')
            logger.info(
                f"      Speedup factor: {cold_time / warm_time if warm_time > 0 else 'N/A'}x"
                )
            logger.info('    Optimized implementation (cold cache)...')
            optimized_analyzer.cache_manager.clear()
            tracemalloc.start()
            start_time = time.time()
            for i in range(iterations):
                with tracer.start_span(
                    f'optimized_confluence_{signal_type}_{signal_direction}_cold'
                    ):
                    result = optimized_analyzer.detect_confluence_optimized(
                        symbol='EURUSD', price_data=optimized_price_data,
                        signal_type=signal_type, signal_direction=
                        signal_direction, related_pairs=related_pairs)
            cold_time = (time.time() - start_time) / iterations
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            results['confluence']['optimized']['cold'].append(cold_time)
            results['memory']['optimized']['confluence'] = peak / 1024 / 1024
            logger.info(
                f'      Time: {cold_time:.4f}s, Memory: {peak / 1024 / 1024:.2f}MB'
                )
            logger.info(
                f"      Confluence score: {result.get('confluence_score', 0):.4f}"
                )
            logger.info(
                f"      Confirmations: {result.get('confirmation_count', 0)}, Contradictions: {result.get('contradiction_count', 0)}"
                )
            logger.info('    Optimized implementation (warm cache)...')
            start_time = time.time()
            for i in range(iterations):
                with tracer.start_span(
                    f'optimized_confluence_{signal_type}_{signal_direction}_warm'
                    ):
                    result = optimized_analyzer.detect_confluence_optimized(
                        symbol='EURUSD', price_data=optimized_price_data,
                        signal_type=signal_type, signal_direction=
                        signal_direction, related_pairs=related_pairs)
            warm_time = (time.time() - start_time) / iterations
            results['confluence']['optimized']['warm'].append(warm_time)
            logger.info(f'      Time: {warm_time:.4f}s')
            logger.info(
                f"      Speedup factor: {cold_time / warm_time if warm_time > 0 else 'N/A'}x"
                )
    logger.info('\nBenchmarking divergence detection...')
    logger.info('  Original implementation (cold cache)...')
    original_analyzer.related_pairs_cache.clear()
    tracemalloc.start()
    start_time = time.time()
    for i in range(iterations):
        with tracer.start_span('original_divergence_cold'):
            result = original_analyzer.analyze_divergence(symbol='EURUSD',
                price_data=price_data, related_pairs=related_pairs)
    cold_time = (time.time() - start_time) / iterations
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results['divergence']['original']['cold'].append(cold_time)
    results['memory']['original']['divergence'] = peak / 1024 / 1024
    logger.info(
        f'    Time: {cold_time:.4f}s, Memory: {peak / 1024 / 1024:.2f}MB')
    logger.info(f"    Divergences found: {result.get('divergences_found', 0)}")
    logger.info('  Original implementation (warm cache)...')
    start_time = time.time()
    for i in range(iterations):
        with tracer.start_span('original_divergence_warm'):
            result = original_analyzer.analyze_divergence(symbol='EURUSD',
                price_data=price_data, related_pairs=related_pairs)
    warm_time = (time.time() - start_time) / iterations
    results['divergence']['original']['warm'].append(warm_time)
    logger.info(f'    Time: {warm_time:.4f}s')
    logger.info(
        f"    Speedup factor: {cold_time / warm_time if warm_time > 0 else 'N/A'}x"
        )
    logger.info('  Optimized implementation (cold cache)...')
    optimized_analyzer.cache_manager.clear()
    tracemalloc.start()
    start_time = time.time()
    for i in range(iterations):
        with tracer.start_span('optimized_divergence_cold'):
            result = optimized_analyzer.analyze_divergence_optimized(symbol
                ='EURUSD', price_data=optimized_price_data, related_pairs=
                related_pairs)
    cold_time = (time.time() - start_time) / iterations
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results['divergence']['optimized']['cold'].append(cold_time)
    results['memory']['optimized']['divergence'] = peak / 1024 / 1024
    logger.info(
        f'    Time: {cold_time:.4f}s, Memory: {peak / 1024 / 1024:.2f}MB')
    logger.info(f"    Divergences found: {result.get('divergences_found', 0)}")
    logger.info('  Optimized implementation (warm cache)...')
    start_time = time.time()
    for i in range(iterations):
        with tracer.start_span('optimized_divergence_warm'):
            result = optimized_analyzer.analyze_divergence_optimized(symbol
                ='EURUSD', price_data=optimized_price_data, related_pairs=
                related_pairs)
    warm_time = (time.time() - start_time) / iterations
    results['divergence']['optimized']['warm'].append(warm_time)
    logger.info(f'    Time: {warm_time:.4f}s')
    logger.info(
        f"    Speedup factor: {cold_time / warm_time if warm_time > 0 else 'N/A'}x"
        )
    logger.info('\nCalculating summary statistics...')
    avg_original_cold_confluence = np.mean(results['confluence']['original'
        ]['cold'])
    avg_original_warm_confluence = np.mean(results['confluence']['original'
        ]['warm'])
    avg_optimized_cold_confluence = np.mean(results['confluence'][
        'optimized']['cold'])
    avg_optimized_warm_confluence = np.mean(results['confluence'][
        'optimized']['warm'])
    avg_original_cold_divergence = np.mean(results['divergence']['original'
        ]['cold'])
    avg_original_warm_divergence = np.mean(results['divergence']['original'
        ]['warm'])
    avg_optimized_cold_divergence = np.mean(results['divergence'][
        'optimized']['cold'])
    avg_optimized_warm_divergence = np.mean(results['divergence'][
        'optimized']['warm'])
    mem_original_confluence = results['memory']['original']['confluence']
    mem_optimized_confluence = results['memory']['optimized']['confluence']
    mem_original_divergence = results['memory']['original']['divergence']
    mem_optimized_divergence = results['memory']['optimized']['divergence']
    confluence_cold_speedup = (avg_original_cold_confluence /
        avg_optimized_cold_confluence)
    confluence_warm_speedup = (avg_original_warm_confluence /
        avg_optimized_warm_confluence)
    confluence_memory_reduction = (mem_original_confluence -
        mem_optimized_confluence) / mem_original_confluence
    divergence_cold_speedup = (avg_original_cold_divergence /
        avg_optimized_cold_divergence)
    divergence_warm_speedup = (avg_original_warm_divergence /
        avg_optimized_warm_divergence)
    divergence_memory_reduction = (mem_original_divergence -
        mem_optimized_divergence) / mem_original_divergence
    logger.info('\nPerformance Summary:')
    logger.info('\nConfluence Detection:')
    logger.info(f'  Original (cold): {avg_original_cold_confluence:.4f}s')
    logger.info(f'  Original (warm): {avg_original_warm_confluence:.4f}s')
    logger.info(f'  Optimized (cold): {avg_optimized_cold_confluence:.4f}s')
    logger.info(f'  Optimized (warm): {avg_optimized_warm_confluence:.4f}s')
    logger.info(f'  Cold Speedup: {confluence_cold_speedup:.2f}x')
    logger.info(f'  Warm Speedup: {confluence_warm_speedup:.2f}x')
    logger.info(
        f'  Memory Usage: {mem_original_confluence:.2f}MB -> {mem_optimized_confluence:.2f}MB ({confluence_memory_reduction:.2%} reduction)'
        )
    logger.info('\nDivergence Detection:')
    logger.info(f'  Original (cold): {avg_original_cold_divergence:.4f}s')
    logger.info(f'  Original (warm): {avg_original_warm_divergence:.4f}s')
    logger.info(f'  Optimized (cold): {avg_optimized_cold_divergence:.4f}s')
    logger.info(f'  Optimized (warm): {avg_optimized_warm_divergence:.4f}s')
    logger.info(f'  Cold Speedup: {divergence_cold_speedup:.2f}x')
    logger.info(f'  Warm Speedup: {divergence_warm_speedup:.2f}x')
    logger.info(
        f'  Memory Usage: {mem_original_divergence:.2f}MB -> {mem_optimized_divergence:.2f}MB ({divergence_memory_reduction:.2%} reduction)'
        )
    logger.info('\nPerformance test completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-world performance test')
    parser.add_argument('--pairs', type=int, default=4, help=
        'Number of currency pairs')
    parser.add_argument('--days', type=int, default=30, help=
        'Number of days of historical data')
    parser.add_argument('--iterations', type=int, default=3, help=
        'Number of iterations for each test')
    parser.add_argument('--gpu', action='store_true', help=
        'Use GPU acceleration')
    args = parser.parse_args()
    asyncio.run(run_performance_test(pairs=args.pairs, days=args.days,
        iterations=args.iterations, use_gpu=args.gpu))
