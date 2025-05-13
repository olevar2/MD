"""
Test script for the optimized confluence and divergence detection algorithms.

This script tests the performance improvements in the RelatedPairsConfluenceAnalyzer
and compares the results with the original implementation.
"""
import asyncio
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from analysis_engine.multi_asset.related_pairs_confluence_detector import RelatedPairsConfluenceAnalyzer
    print('Successfully imported RelatedPairsConfluenceAnalyzer')
except ImportError as e:
    print(f'Error importing RelatedPairsConfluenceAnalyzer: {e}')
    print('Trying alternative import path...')
    try:
        sys.path.insert(0, 'D:\\MD\\forex_trading_platform')
        from analysis_engine.multi_asset.related_pairs_confluence_detector import RelatedPairsConfluenceAnalyzer
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


async def test_performance():
    """Test the performance of the optimized code."""
    print('Creating RelatedPairsConfluenceAnalyzer instance...')
    correlation_service = MockCorrelationService()
    currency_strength_analyzer = MockCurrencyStrengthAnalyzer()
    analyzer = RelatedPairsConfluenceAnalyzer(correlation_service=
        correlation_service, currency_strength_analyzer=
        currency_strength_analyzer, correlation_threshold=0.7,
        lookback_periods=20, cache_ttl_minutes=60, max_workers=4)
    print('Generating sample data...')
    price_data = generate_sample_data(num_pairs=8, num_bars=500)
    print('\nTesting find_related_pairs...')
    start_time = time.time()
    related_pairs = await analyzer.find_related_pairs('EURUSD')
    first_run_time = time.time() - start_time
    print(f'  First run: {first_run_time:.4f} seconds')
    print(f'  Found {len(related_pairs)} related pairs')
    start_time = time.time()
    related_pairs = await analyzer.find_related_pairs('EURUSD')
    second_run_time = time.time() - start_time
    print(f'  Second run (warm cache): {second_run_time:.4f} seconds')
    print(
        f"  Speedup factor: {first_run_time / second_run_time if second_run_time > 0 else 'N/A'}x"
        )
    print('\nTesting detect_confluence...')
    signal_types = ['trend', 'reversal', 'breakout']
    signal_directions = ['bullish', 'bearish']
    for signal_type in signal_types:
        for signal_direction in signal_directions:
            print(f'\n  Testing {signal_type} {signal_direction}...')
            start_time = time.time()
            result = analyzer.detect_confluence(symbol='EURUSD', price_data
                =price_data, signal_type=signal_type, signal_direction=
                signal_direction, related_pairs=related_pairs)
            first_run_time = time.time() - start_time
            print(f'    First run: {first_run_time:.4f} seconds')
            print(
                f"    Confluence score: {result.get('confluence_score', 0):.4f}"
                )
            print(
                f"    Confirmations: {result.get('confirmation_count', 0)}, Contradictions: {result.get('contradiction_count', 0)}"
                )
            start_time = time.time()
            result = analyzer.detect_confluence(symbol='EURUSD', price_data
                =price_data, signal_type=signal_type, signal_direction=
                signal_direction, related_pairs=related_pairs)
            second_run_time = time.time() - start_time
            print(f'    Second run (warm cache): {second_run_time:.4f} seconds'
                )
            print(
                f"    Speedup factor: {first_run_time / second_run_time if second_run_time > 0 else 'N/A'}x"
                )
    print('\nTesting analyze_divergence...')
    start_time = time.time()
    result = analyzer.analyze_divergence(symbol='EURUSD', price_data=
        price_data, related_pairs=related_pairs)
    first_run_time = time.time() - start_time
    print(f'  First run: {first_run_time:.4f} seconds')
    print(f"  Divergences found: {result.get('divergences_found', 0)}")
    start_time = time.time()
    result = analyzer.analyze_divergence(symbol='EURUSD', price_data=
        price_data, related_pairs=related_pairs)
    second_run_time = time.time() - start_time
    print(f'  Second run (warm cache): {second_run_time:.4f} seconds')
    print(
        f"  Speedup factor: {first_run_time / second_run_time if second_run_time > 0 else 'N/A'}x"
        )
    print('\nPerformance test completed!')


if __name__ == '__main__':
    print('Starting test of optimized RelatedPairsConfluenceAnalyzer...')
    asyncio.run(test_performance())
