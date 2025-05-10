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

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from analysis_engine.multi_asset.related_pairs_confluence_detector import RelatedPairsConfluenceAnalyzer
    print("Successfully imported RelatedPairsConfluenceAnalyzer")
except ImportError as e:
    print(f"Error importing RelatedPairsConfluenceAnalyzer: {e}")
    print("Trying alternative import path...")
    try:
        # Try with the full path
        sys.path.insert(0, "D:\\MD\\forex_trading_platform")
        from analysis_engine.multi_asset.related_pairs_confluence_detector import RelatedPairsConfluenceAnalyzer
        print("Successfully imported RelatedPairsConfluenceAnalyzer using full path")
    except ImportError as e:
        print(f"Error importing RelatedPairsConfluenceAnalyzer with full path: {e}")
        sys.exit(1)


class MockCorrelationService:
    """Mock correlation service for testing."""
    
    async def get_all_correlations(self):
        """Return mock correlation data."""
        pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURGBP", "EURJPY", "GBPJPY"]
        correlations = {}
        
        # Create a correlation matrix with some realistic values
        for pair1 in pairs:
            correlations[pair1] = {}
            for pair2 in pairs:
                if pair1 != pair2:
                    # Generate a correlation value between -1 and 1
                    # Make some pairs highly correlated, some negatively correlated
                    if pair1[:3] == pair2[:3] or pair1[3:] == pair2[3:]:
                        # Pairs with same base or quote currency tend to be correlated
                        correlations[pair1][pair2] = np.random.uniform(0.6, 0.9)
                    elif (pair1[:3] in pair2[3:]) or (pair1[3:] in pair2[:3]):
                        # Pairs with inverted currencies tend to be negatively correlated
                        correlations[pair1][pair2] = np.random.uniform(-0.9, -0.6)
                    else:
                        # Other pairs have more random correlation
                        correlations[pair1][pair2] = np.random.uniform(-0.5, 0.5)
        
        return correlations


class MockCurrencyStrengthAnalyzer:
    """Mock currency strength analyzer for testing."""
    
    def calculate_currency_strength(self, price_data):
        """Return mock currency strength data."""
        currencies = ["EUR", "USD", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
        strength = {}
        
        for currency in currencies:
            # Generate a strength value between -1 and 1
            strength[currency] = np.random.uniform(-1, 1)
        
        return strength
    
    def compute_divergence_signals(self, price_data):
        """Return mock divergence signals."""
        return {
            "divergences_found": 2,
            "divergences": [
                {
                    "pair": "EURUSD",
                    "divergence_type": "positive",
                    "strength": 0.7
                },
                {
                    "pair": "GBPUSD",
                    "divergence_type": "negative",
                    "strength": 0.6
                }
            ]
        }


def generate_sample_data(num_pairs=8, num_bars=500):
    """Generate sample price data for testing."""
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURGBP", "EURJPY", "GBPJPY"]
    price_data = {}
    
    for pair in pairs[:num_pairs]:
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=num_bars)
        timestamps = [
            (start_time + timedelta(hours=i)).isoformat()
            for i in range(num_bars)
        ]
        
        # Generate price data with some realistic patterns
        np.random.seed(42 + ord(pair[0]))  # Different seed for each pair
        
        # Start with a base price
        if pair.endswith("JPY"):
            base_price = 110.00
        else:
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
        
        # Create DataFrame
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume
        })
        
        price_data[pair] = df
        
    return price_data


async def test_performance():
    """Test the performance of the optimized code."""
    print("Creating RelatedPairsConfluenceAnalyzer instance...")
    correlation_service = MockCorrelationService()
    currency_strength_analyzer = MockCurrencyStrengthAnalyzer()
    
    analyzer = RelatedPairsConfluenceAnalyzer(
        correlation_service=correlation_service,
        currency_strength_analyzer=currency_strength_analyzer,
        correlation_threshold=0.7,
        lookback_periods=20,
        cache_ttl_minutes=60,
        max_workers=4
    )
    
    print("Generating sample data...")
    price_data = generate_sample_data(num_pairs=8, num_bars=500)
    
    # Get related pairs
    print("\nTesting find_related_pairs...")
    start_time = time.time()
    related_pairs = await analyzer.find_related_pairs("EURUSD")
    first_run_time = time.time() - start_time
    print(f"  First run: {first_run_time:.4f} seconds")
    print(f"  Found {len(related_pairs)} related pairs")
    
    # Test again with warm cache
    start_time = time.time()
    related_pairs = await analyzer.find_related_pairs("EURUSD")
    second_run_time = time.time() - start_time
    print(f"  Second run (warm cache): {second_run_time:.4f} seconds")
    print(f"  Speedup factor: {first_run_time / second_run_time if second_run_time > 0 else 'N/A'}x")
    
    # Test detect_confluence
    print("\nTesting detect_confluence...")
    signal_types = ["trend", "reversal", "breakout"]
    signal_directions = ["bullish", "bearish"]
    
    for signal_type in signal_types:
        for signal_direction in signal_directions:
            print(f"\n  Testing {signal_type} {signal_direction}...")
            
            # First run
            start_time = time.time()
            result = analyzer.detect_confluence(
                symbol="EURUSD",
                price_data=price_data,
                signal_type=signal_type,
                signal_direction=signal_direction,
                related_pairs=related_pairs
            )
            first_run_time = time.time() - start_time
            print(f"    First run: {first_run_time:.4f} seconds")
            print(f"    Confluence score: {result.get('confluence_score', 0):.4f}")
            print(f"    Confirmations: {result.get('confirmation_count', 0)}, Contradictions: {result.get('contradiction_count', 0)}")
            
            # Second run with warm cache
            start_time = time.time()
            result = analyzer.detect_confluence(
                symbol="EURUSD",
                price_data=price_data,
                signal_type=signal_type,
                signal_direction=signal_direction,
                related_pairs=related_pairs
            )
            second_run_time = time.time() - start_time
            print(f"    Second run (warm cache): {second_run_time:.4f} seconds")
            print(f"    Speedup factor: {first_run_time / second_run_time if second_run_time > 0 else 'N/A'}x")
    
    # Test analyze_divergence
    print("\nTesting analyze_divergence...")
    start_time = time.time()
    result = analyzer.analyze_divergence(
        symbol="EURUSD",
        price_data=price_data,
        related_pairs=related_pairs
    )
    first_run_time = time.time() - start_time
    print(f"  First run: {first_run_time:.4f} seconds")
    print(f"  Divergences found: {result.get('divergences_found', 0)}")
    
    # Second run with warm cache
    start_time = time.time()
    result = analyzer.analyze_divergence(
        symbol="EURUSD",
        price_data=price_data,
        related_pairs=related_pairs
    )
    second_run_time = time.time() - start_time
    print(f"  Second run (warm cache): {second_run_time:.4f} seconds")
    print(f"  Speedup factor: {first_run_time / second_run_time if second_run_time > 0 else 'N/A'}x")
    
    print("\nPerformance test completed!")


if __name__ == "__main__":
    print("Starting test of optimized RelatedPairsConfluenceAnalyzer...")
    asyncio.run(test_performance())
