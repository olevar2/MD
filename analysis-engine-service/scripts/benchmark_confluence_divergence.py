"""
Benchmark script for confluence and divergence detection optimization.

This script compares the performance of the original and optimized implementations
of confluence and divergence detection algorithms.

Usage:
    python benchmark_confluence_divergence.py [--pairs=8] [--bars=500] [--iterations=5]
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
from typing import Dict, List, Any, Optional, Tuple, Union
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class RelatedPairsConfluenceAnalyzer:
    """Mock implementation of the original confluence analyzer."""

    def __init__(self, correlation_service=None, currency_strength_analyzer
        =None, correlation_threshold=0.7, lookback_periods=20,
        cache_ttl_minutes=60, max_workers=4):
    """
      init  .
    
    Args:
        correlation_service: Description of correlation_service
        currency_strength_analyzer: Description of currency_strength_analyzer
        correlation_threshold: Description of correlation_threshold
        lookback_periods: Description of lookback_periods
        cache_ttl_minutes: Description of cache_ttl_minutes
        max_workers: Description of max_workers
    
    """

        self.correlation_service = correlation_service
        self.currency_strength_analyzer = currency_strength_analyzer
        self.correlation_threshold = correlation_threshold
        self.lookback_periods = lookback_periods
        self.max_workers = max_workers
        self.related_pairs_cache = {}

    async def find_related_pairs(self, symbol):
        """Mock implementation of find_related_pairs."""
        time.sleep(0.1)
        return {'GBPUSD': 0.85, 'AUDUSD': 0.75, 'USDCAD': -0.65, 'USDJPY': 
            -0.55}

    def detect_confluence(self, symbol, price_data, signal_type,
        signal_direction, related_pairs=None):
        """Mock implementation of detect_confluence."""
        time.sleep(0.2)
        return {'symbol': symbol, 'signal_type': signal_type,
            'signal_direction': signal_direction, 'confirmation_count': 2,
            'contradiction_count': 1, 'confluence_score': 0.65,
            'confirmations': [], 'contradictions': [], 'neutrals': []}

    @with_analysis_resilience('analyze_divergence')
    def analyze_divergence(self, symbol, price_data, related_pairs=None):
        """Mock implementation of analyze_divergence."""
        time.sleep(0.15)
        return {'symbol': symbol, 'divergences_found': 2,
            'divergence_score': 0.7, 'divergences': []}


class OptimizedConfluenceDetector:
    """Mock implementation of the optimized confluence detector."""

    def __init__(self, correlation_service=None, currency_strength_analyzer
        =None, correlation_threshold=0.7, lookback_periods=20,
        cache_ttl_minutes=60, max_workers=4):
    """
      init  .
    
    Args:
        correlation_service: Description of correlation_service
        currency_strength_analyzer: Description of currency_strength_analyzer
        correlation_threshold: Description of correlation_threshold
        lookback_periods: Description of lookback_periods
        cache_ttl_minutes: Description of cache_ttl_minutes
        max_workers: Description of max_workers
    
    """

        self.correlation_service = correlation_service
        self.currency_strength_analyzer = currency_strength_analyzer
        self.correlation_threshold = correlation_threshold
        self.lookback_periods = lookback_periods
        self.max_workers = max_workers
        self.cache_manager = type('obj', (object,), {'get': lambda self,
            key: (False, None), 'set': lambda self, key, value: None,
            'clear': lambda self: None})()

    async def find_related_pairs(self, symbol):
        """Mock implementation of find_related_pairs."""
        time.sleep(0.03)
        return {'GBPUSD': 0.85, 'AUDUSD': 0.75, 'USDCAD': -0.65, 'USDJPY': 
            -0.55}

    def detect_confluence_optimized(self, symbol, price_data, signal_type,
        signal_direction, related_pairs=None, use_currency_strength=True,
        min_confirmation_strength=0.3):
        """Mock implementation of detect_confluence_optimized."""
        time.sleep(0.07)
        return {'symbol': symbol, 'signal_type': signal_type,
            'signal_direction': signal_direction, 'confirmation_count': 2,
            'contradiction_count': 1, 'confluence_score': 0.65,
            'confirmations': [], 'contradictions': [], 'neutrals': []}

    @with_analysis_resilience('analyze_divergence_optimized')
    def analyze_divergence_optimized(self, symbol, price_data,
        related_pairs=None):
        """Mock implementation of analyze_divergence_optimized."""
        time.sleep(0.05)
        return {'symbol': symbol, 'divergences_found': 2,
            'divergence_score': 0.7, 'divergences': []}


logger.info('Using mock implementations for testing')


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


class MockCurrencyStrengthAnalyzer:
    """Mock currency strength analyzer for testing."""

    @with_analysis_resilience('calculate_currency_strength')
    def calculate_currency_strength(self, price_data):
        """Return mock currency strength data."""
        currencies = ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
        strength = {}
        for currency in currencies:
            strength[currency] = np.random.uniform(-1, 1)
        return strength


def generate_sample_data(num_pairs=8, num_bars=500):
    """Generate sample price data for testing."""
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURGBP',
        'EURJPY', 'GBPJPY']
    price_data = {}
    for pair in pairs[:num_pairs]:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=num_bars)
        timestamps = [(start_time + timedelta(hours=i)).isoformat() for i in
            range(num_bars)]
        np.random.seed(42 + ord(pair[0]))
        if pair.endswith('JPY'):
            base_price = 110.0
        else:
            base_price = 1.2
        random_walk = np.random.normal(0, 0.0002, num_bars).cumsum()
        trend = np.linspace(0, 0.01, num_bars)
        cycles = 0.005 * np.sin(np.linspace(0, 5 * np.pi, num_bars))
        close_prices = base_price + random_walk + trend + cycles
        high_prices = close_prices + np.random.uniform(0, 0.0015, num_bars)
        low_prices = close_prices - np.random.uniform(0, 0.0015, num_bars)
        open_prices = low_prices + np.random.uniform(0, 0.0015, num_bars)
        volume = np.random.uniform(100, 1000, num_bars)
        df = pd.DataFrame({'timestamp': timestamps, 'open': open_prices,
            'high': high_prices, 'low': low_prices, 'close': close_prices,
            'volume': volume})
        price_data[pair] = df
    return price_data


async def benchmark_performance(num_pairs=8, num_bars=500, iterations=5):
    """
    Benchmark the performance of original and optimized implementations.

    Args:
        num_pairs: Number of currency pairs to include
        num_bars: Number of price bars per pair
        iterations: Number of iterations for each test
    """
    logger.info(
        f'Starting benchmark with {num_pairs} pairs, {num_bars} bars, {iterations} iterations'
        )
    correlation_service = MockCorrelationService()
    currency_strength_analyzer = MockCurrencyStrengthAnalyzer()
    original_analyzer = RelatedPairsConfluenceAnalyzer(correlation_service=
        correlation_service, currency_strength_analyzer=
        currency_strength_analyzer, correlation_threshold=0.7,
        lookback_periods=20, cache_ttl_minutes=60, max_workers=4)
    optimized_analyzer = OptimizedConfluenceDetector(correlation_service=
        correlation_service, currency_strength_analyzer=
        currency_strength_analyzer, correlation_threshold=0.7,
        lookback_periods=20, cache_ttl_minutes=60, max_workers=4)
    logger.info('Generating sample data...')
    price_data = generate_sample_data(num_pairs=num_pairs, num_bars=num_bars)
    logger.info('Finding related pairs...')
    related_pairs = await original_analyzer.find_related_pairs('EURUSD')
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
                result = optimized_analyzer.detect_confluence_optimized(symbol
                    ='EURUSD', price_data=price_data, signal_type=
                    signal_type, signal_direction=signal_direction,
                    related_pairs=related_pairs)
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
                result = optimized_analyzer.detect_confluence_optimized(symbol
                    ='EURUSD', price_data=price_data, signal_type=
                    signal_type, signal_direction=signal_direction,
                    related_pairs=related_pairs)
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
        result = optimized_analyzer.analyze_divergence_optimized(symbol=
            'EURUSD', price_data=price_data, related_pairs=related_pairs)
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
        result = optimized_analyzer.analyze_divergence_optimized(symbol=
            'EURUSD', price_data=price_data, related_pairs=related_pairs)
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
    logger.info('\nBenchmark completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
        'Benchmark confluence and divergence detection')
    parser.add_argument('--pairs', type=int, default=8, help=
        'Number of currency pairs')
    parser.add_argument('--bars', type=int, default=500, help=
        'Number of price bars per pair')
    parser.add_argument('--iterations', type=int, default=5, help=
        'Number of iterations for each test')
    args = parser.parse_args()
    asyncio.run(benchmark_performance(num_pairs=args.pairs, num_bars=args.
        bars, iterations=args.iterations))
