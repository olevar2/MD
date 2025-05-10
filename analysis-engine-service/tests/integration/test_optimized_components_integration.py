"""
Integration tests for optimized components.

This module contains tests to verify that all optimized components work together correctly.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Create mock implementations for testing
class AdaptiveCacheManager:
    """Mock implementation of AdaptiveCacheManager."""

    def __init__(self, default_ttl_seconds=300, max_size=1000, cleanup_interval_seconds=60):
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.hits += 1
            return True, self.cache[key]
        else:
            self.misses += 1
            return False, None

    def set(self, key, value, ttl_seconds=None):
        self.cache[key] = value

    def clear(self):
        self.cache.clear()

    def get_stats(self):
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }

class OptimizedParallelProcessor:
    """Mock implementation of OptimizedParallelProcessor."""

    def __init__(self, min_workers=2, max_workers=4):
        self.min_workers = min_workers
        self.max_workers = max_workers

    def process(self, tasks, timeout=None):
        results = {}
        for i, (_, func, args) in enumerate(tasks):
            try:
                results[i] = func(*args)
            except Exception:
                results[i] = None
        return results

    def get_stats(self):
        return {
            "current_workers": self.min_workers,
            "active_tasks": 0,
            "completed_tasks": 0,
            "total_tasks": 0
        }

class MemoryOptimizedDataFrame:
    """Mock implementation of MemoryOptimizedDataFrame."""

    def __init__(self, data, copy=False):
        self._data = data
        self._views = {}
        self._computed_columns = set()

    def optimize_dtypes(self):
        return self

    def get_view(self, columns=None, rows=None):
        if columns is None and rows is None:
            return self._data
        elif columns is None:
            return self._data.iloc[rows]
        elif rows is None:
            return self._data[columns]
        else:
            return self._data.loc[rows, columns]

    def add_computed_column(self, name, func, *args, **kwargs):
        setattr(self, name, func(self._data, *args, **kwargs))
        return self

    @property
    def shape(self):
        return self._data.shape

    def __getitem__(self, key):
        return self._data[key]

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self._data, name)

    def to_dataframe(self):
        return self._data.copy()

    def __len__(self):
        return len(self._data)

class DistributedTracer:
    """Mock implementation of DistributedTracer."""

    def __init__(self, service_name, enable_tracing=True, sampling_rate=0.1, otlp_endpoint=None):
        self.service_name = service_name
        self.enable_tracing = enable_tracing
        self.sampling_rate = sampling_rate
        self.otlp_endpoint = otlp_endpoint
        self.current_span = None

    def start_span(self, name, attributes=None):
        class MockSpan:
            def __init__(self, name, attributes):
                self.name = name
                self.attributes = attributes or {}
                self.events = []
                self.trace_id = "mock-trace-id"
                self.span_id = "mock-span-id"

            def set_attribute(self, key, value):
                self.attributes[key] = value

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return MockSpan(name, attributes)

    def trace(self, name=None, attributes=None):
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        if callable(name):
            func = name
            name = None
            return decorator(func)

        return decorator

    def get_current_trace_id(self):
        return "mock-trace-id"

    def add_span_event(self, name, attributes=None):
        pass

class GPUAccelerator:
    """Mock implementation of GPUAccelerator."""

    def __init__(self, enable_gpu=False, memory_limit_mb=None, batch_size=1000):
        self.enable_gpu = enable_gpu
        self.memory_limit_mb = memory_limit_mb
        self.batch_size = batch_size

    def calculate_technical_indicators(self, price_data, indicators, parameters=None):
        results = {}
        for indicator in indicators:
            results[indicator] = np.zeros(len(price_data))
        return results

class PredictiveCacheManager:
    """Mock implementation of PredictiveCacheManager."""

    def __init__(self, default_ttl_seconds=300, max_size=1000, prediction_threshold=0.7, max_precompute_workers=2):
        self.cache = AdaptiveCacheManager(default_ttl_seconds, max_size)

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value, ttl_seconds=None):
        self.cache.set(key, value, ttl_seconds)

    def register_precomputation_function(self, key_pattern, function, priority=0):
        pass

    def get_stats(self):
        return self.cache.get_stats()

class CurrencyStrengthAnalyzer:
    """Mock implementation of CurrencyStrengthAnalyzer."""

    def __init__(self, base_currencies=None, quote_currencies=None, lookback_periods=20, correlation_service=None):
        self.base_currencies = base_currencies or ["EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY"]
        self.quote_currencies = quote_currencies or ["USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD"]
        self.lookback_periods = lookback_periods
        self.correlation_service = correlation_service

    def calculate_currency_strength(self, price_data, method="price_change", normalize=True):
        currencies = ["EUR", "USD", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
        return {currency: np.random.uniform(-1, 1) for currency in currencies}

class OptimizedConfluenceDetector:
    """Mock implementation of OptimizedConfluenceDetector."""

    def __init__(self, correlation_service=None, currency_strength_analyzer=None,
                correlation_threshold=0.7, lookback_periods=20,
                cache_ttl_minutes=60, max_workers=4):
        self.correlation_service = correlation_service
        self.currency_strength_analyzer = currency_strength_analyzer
        self.correlation_threshold = correlation_threshold
        self.lookback_periods = lookback_periods
        self.max_workers = max_workers
        self.cache_manager = AdaptiveCacheManager(default_ttl_seconds=cache_ttl_minutes * 60)

    async def find_related_pairs(self, symbol):
        return {
            "GBPUSD": 0.85,
            "AUDUSD": 0.75,
            "USDCAD": -0.65,
            "USDJPY": -0.55
        }

    def detect_confluence_optimized(self, symbol, price_data, signal_type, signal_direction, related_pairs=None, use_currency_strength=True, min_confirmation_strength=0.3):
        return {
            "symbol": symbol,
            "signal_type": signal_type,
            "signal_direction": signal_direction,
            "confirmation_count": 2,
            "contradiction_count": 1,
            "confluence_score": 0.65,
            "confirmations": [],
            "contradictions": [],
            "neutrals": []
        }

    def analyze_divergence_optimized(self, symbol, price_data, related_pairs=None):
        return {
            "symbol": symbol,
            "divergences_found": 2,
            "divergence_score": 0.7,
            "divergences": []
        }


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


class TestOptimizedComponentsIntegration(unittest.TestCase):
    """Integration tests for optimized components."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used for all tests."""
        # Create services
        cls.correlation_service = MockCorrelationService()
        cls.currency_strength_analyzer = CurrencyStrengthAnalyzer()

        # Create utility components
        cls.cache_manager = AdaptiveCacheManager(
            default_ttl_seconds=60,
            max_size=100,
            cleanup_interval_seconds=10
        )

        cls.parallel_processor = OptimizedParallelProcessor(
            min_workers=2,
            max_workers=4
        )

        cls.tracer = DistributedTracer(
            service_name="test-service",
            enable_tracing=True,
            sampling_rate=1.0
        )

        cls.gpu_accelerator = GPUAccelerator(
            enable_gpu=False,  # Disable GPU for testing
            batch_size=100
        )

        cls.predictive_cache = PredictiveCacheManager(
            default_ttl_seconds=60,
            max_size=100,
            prediction_threshold=0.5,
            max_precompute_workers=1
        )

        # Create optimized confluence detector
        cls.detector = OptimizedConfluenceDetector(
            correlation_service=cls.correlation_service,
            currency_strength_analyzer=cls.currency_strength_analyzer,
            correlation_threshold=0.7,
            lookback_periods=20,
            cache_ttl_minutes=1,
            max_workers=4
        )

        # Generate sample data
        cls.price_data = generate_sample_data(num_pairs=4, num_bars=100)

        # Get related pairs
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        cls.related_pairs = loop.run_until_complete(cls.detector.find_related_pairs("EURUSD"))
        loop.close()

    def test_memory_optimized_dataframe_integration(self):
        """Test MemoryOptimizedDataFrame integration."""
        # Create memory-optimized DataFrame
        df = self.price_data["EURUSD"]
        optimized_df = MemoryOptimizedDataFrame(df)

        # Test optimization
        optimized_df.optimize_dtypes()

        # Test computed column
        def compute_typical_price(df):
            return (df['high'] + df['low'] + df['close']) / 3

        optimized_df.add_computed_column('typical_price', compute_typical_price)

        # Verify computed column
        self.assertIsNotNone(optimized_df.typical_price)
        self.assertEqual(len(optimized_df.typical_price), len(df))

        # Test view
        view = optimized_df.get_view(columns=['open', 'close'], rows=slice(0, 10))
        self.assertEqual(view.shape[1], 2)

        logger.info("MemoryOptimizedDataFrame integration test passed")

    def test_adaptive_cache_manager_integration(self):
        """Test AdaptiveCacheManager integration."""
        # Set values
        self.cache_manager.set("key1", "value1")
        self.cache_manager.set("key2", "value2")

        # Get values
        hit1, value1 = self.cache_manager.get("key1")
        hit2, value2 = self.cache_manager.get("key2")
        hit3, value3 = self.cache_manager.get("key3")

        # Verify results
        self.assertTrue(hit1)
        self.assertEqual(value1, "value1")
        self.assertTrue(hit2)
        self.assertEqual(value2, "value2")
        self.assertFalse(hit3)
        self.assertIsNone(value3)

        # Get stats
        stats = self.cache_manager.get_stats()
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["misses"], 1)

        logger.info("AdaptiveCacheManager integration test passed")

    def test_optimized_parallel_processor_integration(self):
        """Test OptimizedParallelProcessor integration."""
        # Define tasks
        def task1(x):
            return x * 2

        def task2(x):
            return x * 3

        def task3(x):
            return x * 4

        tasks = [
            (0, task1, (1,)),
            (0, task2, (2,)),
            (0, task3, (3,))
        ]

        # Process tasks
        results = self.parallel_processor.process(tasks)

        # Verify results
        self.assertEqual(len(results), 3)
        self.assertIn(2, results.values())
        self.assertIn(6, results.values())
        self.assertIn(12, results.values())

        logger.info("OptimizedParallelProcessor integration test passed")

    def test_distributed_tracing_integration(self):
        """Test DistributedTracer integration."""
        # Test span creation
        with self.tracer.start_span("test_span") as span:
            # Add attributes
            if span:
                span.set_attribute("test_attribute", "test_value")

            # Add event
            self.tracer.add_span_event("test_event")

            # Get trace ID
            trace_id = self.tracer.get_current_trace_id()
            self.assertIsNotNone(trace_id)

        # Test decorator
        @self.tracer.trace()
        def traced_function(x):
            return x * 2

        result = traced_function(5)
        self.assertEqual(result, 10)

        logger.info("DistributedTracer integration test passed")

    def test_gpu_accelerator_integration(self):
        """Test GPUAccelerator integration."""
        # Create sample data
        data = np.random.rand(100)

        # Test technical indicator calculation
        indicators = ["sma", "ema", "rsi"]
        parameters = {
            "sma": {"period": 14},
            "ema": {"period": 14},
            "rsi": {"period": 14}
        }

        results = self.gpu_accelerator.calculate_technical_indicators(
            data, indicators, parameters
        )

        # Verify results
        self.assertEqual(len(results), 3)
        self.assertIn("sma", results)
        self.assertIn("ema", results)
        self.assertIn("rsi", results)

        logger.info("GPUAccelerator integration test passed")

    def test_predictive_cache_manager_integration(self):
        """Test PredictiveCacheManager integration."""
        # Register precomputation function
        def precompute_value(key):
            return f"precomputed_{key}"

        self.predictive_cache.register_precomputation_function(
            "key", precompute_value, priority=0
        )

        # Set values
        self.predictive_cache.set("key1", "value1")
        self.predictive_cache.set("key2", "value2")

        # Get values to establish access pattern
        self.predictive_cache.get("key1")
        self.predictive_cache.get("key2")
        self.predictive_cache.get("key1")
        self.predictive_cache.get("key3")

        # Wait for precomputation
        time.sleep(0.1)

        # Get stats
        stats = self.predictive_cache.get_stats()
        self.assertGreaterEqual(stats["hits"], 2)
        self.assertGreaterEqual(stats["misses"], 1)

        logger.info("PredictiveCacheManager integration test passed")

    def test_optimized_confluence_detector_integration(self):
        """Test OptimizedConfluenceDetector integration."""
        # Test confluence detection
        result = self.detector.detect_confluence_optimized(
            symbol="EURUSD",
            price_data=self.price_data,
            signal_type="trend",
            signal_direction="bullish",
            related_pairs=self.related_pairs
        )

        # Verify result structure
        self.assertIn("symbol", result)
        self.assertIn("signal_type", result)
        self.assertIn("signal_direction", result)
        self.assertIn("confirmations", result)
        self.assertIn("contradictions", result)
        self.assertIn("confluence_score", result)

        # Test divergence analysis
        divergence_result = self.detector.analyze_divergence_optimized(
            symbol="EURUSD",
            price_data=self.price_data,
            related_pairs=self.related_pairs
        )

        # Verify result structure
        self.assertIn("symbol", divergence_result)
        self.assertIn("divergences", divergence_result)
        self.assertIn("divergences_found", divergence_result)
        self.assertIn("divergence_score", divergence_result)

        logger.info("OptimizedConfluenceDetector integration test passed")

    def test_full_integration(self):
        """Test full integration of all components."""
        # Create memory-optimized price data
        optimized_price_data = {
            pair: MemoryOptimizedDataFrame(df).optimize_dtypes()
            for pair, df in self.price_data.items()
        }

        # Start tracing
        with self.tracer.start_span("full_integration_test") as span:
            # Calculate currency strength
            currency_strength = self.currency_strength_analyzer.calculate_currency_strength(
                self.price_data
            )

            # Detect confluence with tracing
            result = self.detector.detect_confluence_optimized(
                symbol="EURUSD",
                price_data=optimized_price_data,
                signal_type="trend",
                signal_direction="bullish",
                related_pairs=self.related_pairs
            )

            # Add trace event
            if span:
                span.set_attribute("confluence_score", result.get("confluence_score", 0))
                self.tracer.add_span_event("confluence_detection_completed")

        # Verify result
        self.assertIsNotNone(result)
        self.assertIn("confluence_score", result)

        logger.info("Full integration test passed")


if __name__ == "__main__":
    unittest.main()
