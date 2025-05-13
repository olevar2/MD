"""
Profile the performance of the RelatedPairsConfluenceAnalyzer.

This script profiles the performance of the RelatedPairsConfluenceAnalyzer class
and identifies bottlenecks in the confluence and divergence detection algorithms.
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
from typing import Dict, List, Any, Optional, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from analysis_engine.multi_asset.related_pairs_confluence_detector import RelatedPairsConfluenceAnalyzer
    from analysis_engine.multi_asset.correlation_tracking_service import CorrelationTrackingService
    from analysis_engine.multi_asset.currency_strength_analyzer import CurrencyStrengthAnalyzer
    print('Successfully imported RelatedPairsConfluenceAnalyzer')
except ImportError as e:
    print(f'Error importing RelatedPairsConfluenceAnalyzer: {e}')
    print('Trying alternative import path...')
    try:
        sys.path.insert(0, 'D:\\MD\\forex_trading_platform')
        from analysis_engine.multi_asset.related_pairs_confluence_detector import RelatedPairsConfluenceAnalyzer
        from analysis_engine.multi_asset.correlation_tracking_service import CorrelationTrackingService
        from analysis_engine.multi_asset.currency_strength_analyzer import CurrencyStrengthAnalyzer
        print(
            'Successfully imported RelatedPairsConfluenceAnalyzer using full path'
            )
    except ImportError as e:
        print(
            f'Error importing RelatedPairsConfluenceAnalyzer with full path: {e}'
            )
        sys.exit(1)


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

    def compute_divergence_signals(self, price_data):
        """Return mock divergence signals."""
        return {'divergences_found': 2, 'divergences': [{'pair': 'EURUSD',
            'divergence_type': 'positive', 'strength': 0.7}, {'pair':
            'GBPUSD', 'divergence_type': 'negative', 'strength': 0.6}]}


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


async def profile_find_related_pairs(analyzer, symbol='EURUSD'):
    """Profile the find_related_pairs method."""
    start_time = time.time()
    result1 = await analyzer.find_related_pairs(symbol)
    first_run_time = time.time() - start_time
    start_time = time.time()
    result2 = await analyzer.find_related_pairs(symbol)
    second_run_time = time.time() - start_time
    return {'cold_cache_time': first_run_time, 'warm_cache_time':
        second_run_time, 'speedup_factor': first_run_time / second_run_time if
        second_run_time > 0 else 0, 'related_pairs_count': len(result2)}


def profile_detect_confluence(analyzer, price_data, symbol='EURUSD',
    related_pairs=None):
    """Profile the detect_confluence method."""
    signal_types = ['trend', 'reversal', 'breakout']
    signal_directions = ['bullish', 'bearish']
    results = {}
    for signal_type in signal_types:
        for signal_direction in signal_directions:
            start_time = time.time()
            result1 = analyzer.detect_confluence(symbol=symbol, price_data=
                price_data, signal_type=signal_type, signal_direction=
                signal_direction, related_pairs=related_pairs)
            first_run_time = time.time() - start_time
            start_time = time.time()
            result2 = analyzer.detect_confluence(symbol=symbol, price_data=
                price_data, signal_type=signal_type, signal_direction=
                signal_direction, related_pairs=related_pairs)
            second_run_time = time.time() - start_time
            results[f'{signal_type}_{signal_direction}'] = {'cold_cache_time':
                first_run_time, 'warm_cache_time': second_run_time,
                'speedup_factor': first_run_time / second_run_time if 
                second_run_time > 0 else 0, 'confluence_score': result2.get
                ('confluence_score', 0)}
    return results


def profile_analyze_divergence(analyzer, price_data, symbol='EURUSD',
    related_pairs=None):
    """Profile the analyze_divergence method."""
    start_time = time.time()
    result1 = analyzer.analyze_divergence(symbol=symbol, price_data=
        price_data, related_pairs=related_pairs)
    first_run_time = time.time() - start_time
    start_time = time.time()
    result2 = analyzer.analyze_divergence(symbol=symbol, price_data=
        price_data, related_pairs=related_pairs)
    second_run_time = time.time() - start_time
    return {'cold_cache_time': first_run_time, 'warm_cache_time':
        second_run_time, 'speedup_factor': first_run_time / second_run_time if
        second_run_time > 0 else 0, 'divergences_found': result2.get(
        'divergences_found', 0)}


async def run_profiling():
    """Run the profiling tests."""
    print('Creating mock services...')
    correlation_service = MockCorrelationService()
    currency_strength_analyzer = MockCurrencyStrengthAnalyzer()
    print('Creating RelatedPairsConfluenceAnalyzer instance...')
    analyzer = RelatedPairsConfluenceAnalyzer(correlation_service=
        correlation_service, currency_strength_analyzer=
        currency_strength_analyzer, correlation_threshold=0.7,
        lookback_periods=20, cache_ttl_minutes=60, max_workers=4)
    print('Generating sample data...')
    price_data = generate_sample_data(num_pairs=8, num_bars=500)
    print('\nProfiling find_related_pairs method...')
    related_pairs_results = await profile_find_related_pairs(analyzer,
        symbol='EURUSD')
    print(
        f"  Cold cache: {related_pairs_results['cold_cache_time']:.4f} seconds"
        )
    print(
        f"  Warm cache: {related_pairs_results['warm_cache_time']:.4f} seconds"
        )
    print(f"  Speedup factor: {related_pairs_results['speedup_factor']:.2f}x")
    print(
        f"  Found {related_pairs_results['related_pairs_count']} related pairs"
        )
    related_pairs = await analyzer.find_related_pairs('EURUSD')
    print('\nProfiling detect_confluence method...')
    confluence_results = profile_detect_confluence(analyzer, price_data,
        'EURUSD', related_pairs)
    for key, result in confluence_results.items():
        print(f'  {key}:')
        print(f"    Cold cache: {result['cold_cache_time']:.4f} seconds")
        print(f"    Warm cache: {result['warm_cache_time']:.4f} seconds")
        print(f"    Speedup factor: {result['speedup_factor']:.2f}x")
        print(f"    Confluence score: {result['confluence_score']:.4f}")
    print('\nProfiling analyze_divergence method...')
    divergence_results = profile_analyze_divergence(analyzer, price_data,
        'EURUSD', related_pairs)
    print(f"  Cold cache: {divergence_results['cold_cache_time']:.4f} seconds")
    print(f"  Warm cache: {divergence_results['warm_cache_time']:.4f} seconds")
    print(f"  Speedup factor: {divergence_results['speedup_factor']:.2f}x")
    print(f"  Found {divergence_results['divergences_found']} divergences")
    return {'related_pairs': related_pairs_results, 'confluence':
        confluence_results, 'divergence': divergence_results}


async def main():
    """Main function to run the profiling."""
    print('Profiling RelatedPairsConfluenceAnalyzer performance...')
    results = await run_profiling()
    print('\nProfiling completed!')


if __name__ == '__main__':
    asyncio.run(main())
