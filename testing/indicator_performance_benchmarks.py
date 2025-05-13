"""
Indicator Performance Benchmarking

This module provides benchmarking tools for measuring and optimizing the performance
of technical indicators in the forex trading platform. It includes:

1. Benchmarking for calculation speed of indicators
2. Identification of bottleneck operations
3. Testing of caching strategies for frequently used calculations
4. Memory usage profiling

These benchmarks complete the performance aspect of Phase 7 implementation.
"""

import os
import time
import gc
import cProfile
import pstats
import io
import tracemalloc
from datetime import datetime, timedelta
from functools import wraps
from memory_profiler import profile as memory_profile
from typing import Dict, List, Any, Optional, Union, Callable, Type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

# Import core components
from core_foundations.utils.logger import get_logger
from core_foundations.config.settings import get_settings

# Import feature store components for indicators
from feature_store_service.feature_store_service.indicators.advanced_moving_averages import (
    TripleExponentialMovingAverage, DoubleExponentialMovingAverage, HullMovingAverage,
    KaufmanAdaptiveMovingAverage, ZeroLagExponentialMovingAverage, ArnaudLegouxMovingAverage
)
from feature_store_service.feature_store_service.indicators.advanced_oscillators import (
    AwesomeOscillator, AcceleratorOscillator, UltimateOscillator, DeMarker,
    TRIX, KnowSureThing, ElderForceIndex, RelativeVigorIndex
)
from feature_store_service.feature_store_service.indicators.volume_analysis import (
    VolumeZoneOscillator, EaseOfMovement, NVIAndPVI, RelativeVolume, VolumeDelta
)

# Import multi-timeframe components
from analysis_engine.analysis.multi_timeframe import (
    MultiTimeframeIndicator, TimeframeConfluenceScanner
)

# Import optional caching utilities
try:
    from feature_store_service.feature_store_service.caching import IndicatorCache
    HAS_CACHING = True
except ImportError:
    HAS_CACHING = False

logger = get_logger(__name__)
settings = get_settings()


# Decorator for timing functions
def timing_decorator(func):
    """Decorator to measure execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        logger.info(f"Function {func.__name__} took {execution_time:.6f} seconds to execute")
        return result, execution_time
    return wrapper


# Function to generate sample data
def generate_sample_data(size=5000, random_seed=42) -> pd.DataFrame:
    """Generate sample OHLCV data for benchmarking."""
    np.random.seed(random_seed)
    dates = pd.date_range(start='2024-01-01', periods=size, freq='1H')
    
    # Start with a base price and add random walks with some trend
    base_price = 100.0
    volatility = 0.5
    drift = 0.01
    
    # Generate random returns with drift
    returns = np.random.normal(drift, volatility, size=len(dates)) / 100
    
    # Calculate prices using cumulative returns
    price_path = base_price * (1 + np.cumsum(returns))
    
    # Generate OHLCV data
    high_vals = price_path * (1 + np.random.uniform(0, 0.01, size=len(dates)))
    low_vals = price_path * (1 - np.random.uniform(0, 0.01, size=len(dates)))
    
    # Make sure high is always >= open/close, and low is always <= open/close
    close_vals = price_path
    open_vals = price_path * (1 + np.random.normal(0, 0.003, size=len(dates)))
    
    # Ensure proper OHLC relationships
    for i in range(len(dates)):
        high_vals[i] = max(high_vals[i], open_vals[i], close_vals[i])
        low_vals[i] = min(low_vals[i], open_vals[i], close_vals[i])
    
    # Generate volume data - higher volume on bigger price moves
    volume = np.abs(np.diff(np.append(0, price_path))) * 1000000 + 100000
    volume = volume * np.random.uniform(0.8, 1.2, size=len(dates))
    
    # Create DataFrame
    df = pd.DataFrame({
        'datetime': dates,
        'open': open_vals,
        'high': high_vals,
        'low': low_vals,
        'close': close_vals,
        'volume': volume
    })
    
    df.set_index('datetime', inplace=True)
    return df


class IndicatorPerformanceBenchmark:
    """
    Class for benchmarking indicator performance.
    
    This class provides tools for measuring execution time, memory usage,
    and identifying bottlenecks in indicator calculations.
    """
    
    def __init__(self, data_sizes=None, repeat=3, output_dir=None):
        """
        Initialize the benchmark with configuration parameters.
        
        Args:
            data_sizes: List of data sizes to test with
            repeat: Number of times to repeat each test for reliable results
            output_dir: Directory to store benchmark results
        """
        self.data_sizes = data_sizes or [1000, 5000, 10000, 20000]
        self.repeat = repeat
        self.output_dir = output_dir or os.path.join("output", "benchmarks", 
                                                   f"indicator_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results = {}
        self.memory_results = {}
        
        self.indicators = {
            "TEMA": TripleExponentialMovingAverage(period=20),
            "DEMA": DoubleExponentialMovingAverage(period=20),
            "Hull": HullMovingAverage(period=20),
            "KAMA": KaufmanAdaptiveMovingAverage(period=20, fast_period=2, slow_period=30),
            "ZLEMA": ZeroLagExponentialMovingAverage(period=20),
            "ALMA": ArnaudLegouxMovingAverage(period=20, sigma=6.0, offset=0.85),
            "AO": AwesomeOscillator(),
            "ACC": AcceleratorOscillator(),
            "UltOsc": UltimateOscillator(),
            "DeMarker": DeMarker(),
            "TRIX": TRIX(),
            "KST": KnowSureThing(),
            "RVI": RelativeVigorIndex(),
            "VZO": VolumeZoneOscillator(),
            "EOM": EaseOfMovement(),
            "NVI_PVI": NVIAndPVI(),
            "RelVol": RelativeVolume(),
            "VolDelta": VolumeDelta()
        }
    
    def benchmark_calculation_speed(self, indicator_names=None):
        """
        Benchmark the calculation speed of indicators.
        
        Args:
            indicator_names: List of indicator names to benchmark (all if None)
        """
        if indicator_names is None:
            indicator_names = list(self.indicators.keys())
        
        logger.info(f"Benchmarking calculation speed for {len(indicator_names)} indicators")
        
        # Prepare result container
        self.results = {indicator: {size: [] for size in self.data_sizes} 
                         for indicator in indicator_names}
        
        # Run benchmarks for each data size
        for size in self.data_sizes:
            logger.info(f"Generating sample data with {size} rows")
            data = generate_sample_data(size=size)
            
            # Test each indicator
            for name in indicator_names:
                indicator = self.indicators[name]
                
                # Clear any cached data
                gc.collect()
                
                logger.info(f"Benchmarking {name} with {size} rows")
                
                # Repeat the test for more reliable results
                times = []
                for r in range(self.repeat):
                    start_time = time.perf_counter()
                    indicator.calculate(data)
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    times.append(execution_time)
                
                # Store the average execution time
                avg_time = sum(times) / len(times)
                self.results[name][size] = avg_time
                
                logger.info(f"{name} with {size} rows: {avg_time:.6f} seconds (avg of {self.repeat} runs)")
        
        # Save results
        self._save_results("calculation_speed", self.results)
        
        return self.results
    
    def profile_indicator_calculations(self, indicator_names=None, data_size=10000):
        """
        Profile the indicator calculations to identify bottlenecks.
        
        Args:
            indicator_names: List of indicator names to profile (all if None)
            data_size: Size of the data to use for profiling
            
        Returns:
            Dictionary with profiling results
        """
        if indicator_names is None:
            indicator_names = list(self.indicators.keys())
        
        logger.info(f"Profiling calculations for {len(indicator_names)} indicators with {data_size} rows")
        
        data = generate_sample_data(size=data_size)
        profiling_results = {}
        
        for name in indicator_names:
            indicator = self.indicators[name]
            
            # Clear any cached data
            gc.collect()
            
            logger.info(f"Profiling {name}")
            
            # Use cProfile to get detailed profiling information
            pr = cProfile.Profile()
            pr.enable()
            indicator.calculate(data)
            pr.disable()
            
            # Get stats as string
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Print top 20 functions by cumulative time
            
            profiling_results[name] = s.getvalue()
            
            # Save profiling results to a file
            profile_file = os.path.join(self.output_dir, f"profile_{name}.txt")
            with open(profile_file, "w") as f:
                ps.stream = f
                ps.print_stats(30)  # Save top 30 functions to file
        
        return profiling_results
    
    def benchmark_with_caching(self, indicator_names=None, data_size=10000):
        """
        Benchmark indicators with and without caching.
        
        Args:
            indicator_names: List of indicator names to benchmark (all if None)
            data_size: Size of the data to use for benchmarking
            
        Returns:
            Dictionary with benchmarking results
        """
        if not HAS_CACHING:
            logger.warning("Caching utilities not available. Skipping caching benchmark.")
            return {}
        
        if indicator_names is None:
            indicator_names = list(self.indicators.keys())
        
        logger.info(f"Benchmarking caching for {len(indicator_names)} indicators with {data_size} rows")
        
        data = generate_sample_data(size=data_size)
        caching_results = {
            "without_cache": {},
            "with_cache": {},
            "speedup": {}
        }
        
        # Create a cache instance
        indicator_cache = IndicatorCache(max_size=100)
        
        for name in indicator_names:
            indicator = self.indicators[name]
            
            # Benchmark without caching
            gc.collect()
            logger.info(f"Benchmarking {name} without cache")
            
            times = []
            for r in range(self.repeat):
                start_time = time.perf_counter()
                indicator.calculate(data)
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                times.append(execution_time)
            
            without_cache_time = sum(times) / len(times)
            caching_results["without_cache"][name] = without_cache_time
            
            # Benchmark with caching
            gc.collect()
            logger.info(f"Benchmarking {name} with cache")
            
            # Add the indicator to the cache
            indicator_with_cache = indicator_cache.register_indicator(indicator)
            
            times = []
            for r in range(self.repeat):
                start_time = time.perf_counter()
                indicator_with_cache.calculate(data)
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                times.append(execution_time)
                
                # Clear cache between runs to ensure fair comparison
                if r < self.repeat - 1:
                    indicator_cache.clear()
            
            with_cache_time = sum(times) / len(times)
            caching_results["with_cache"][name] = with_cache_time
            
            # Calculate speedup
            speedup = without_cache_time / with_cache_time if with_cache_time > 0 else float('inf')
            caching_results["speedup"][name] = speedup
            
            logger.info(f"{name}: without cache: {without_cache_time:.6f}s, with cache: {with_cache_time:.6f}s, "
                       f"speedup: {speedup:.2f}x")
        
        # Save results
        self._save_results("caching_benchmark", caching_results)
        
        return caching_results
    
    @timing_decorator
    def benchmark_multi_timeframe_analysis(self, data_size=10000):
        """
        Benchmark multi-timeframe analysis performance.
        
        Args:
            data_size: Size of the base data to use for benchmarking
            
        Returns:
            Result and execution time
        """
        logger.info(f"Benchmarking multi-timeframe analysis with {data_size} base rows")
        
        # Generate sample data
        base_data = generate_sample_data(size=data_size)
        
        # Create different timeframe data
        data_1h = base_data.copy()
        
        # Resample to 4h
        data_4h = data_1h.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Resample to 1d
        data_1d = data_1h.resample('1D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        tf_data = {
            '1h': data_1h,
            '4h': data_4h,
            '1d': data_1d
        }
        
        # Create indicators for multi-timeframe analysis
        tema = TripleExponentialMovingAverage(period=20)
        rvi = RelativeVigorIndex(period=14)
        
        # Create multi-timeframe wrapper
        mtf_tema = MultiTimeframeIndicator(
            indicator=tema,
            timeframes=['1h', '4h', '1d']
        )
        
        mtf_rvi = MultiTimeframeIndicator(
            indicator=rvi,
            timeframes=['1h', '4h', '1d']
        )
        
        # Calculate indicators on multiple timeframes
        mtf_tema_result = mtf_tema.calculate(tf_data)
        mtf_rvi_result = mtf_rvi.calculate(tf_data)
        
        # Create timeframe confluence scanner
        confluence_scanner = TimeframeConfluenceScanner(
            timeframes=['1h', '4h', '1d'],
            indicators=['tema_20', 'rvi_14']
        )
        
        # Combine the results for confluence scanning
        combined_data = {}
        for tf in ['1h', '4h', '1d']:
            combined_data[tf] = pd.DataFrame({
                'close': tf_data[tf]['close'],
                'tema_20': mtf_tema_result[tf]['tema_20'] if 'tema_20' in mtf_tema_result[tf].columns else None,
                'rvi_14': mtf_rvi_result[tf]['rvi_14'] if 'rvi_14' in mtf_rvi_result[tf].columns else None
            })
        
        # Scan for confluence signals
        confluence_signals = confluence_scanner.scan_for_confluence(combined_data)
        
        return confluence_signals
    
    def profile_memory_usage(self, indicator_names=None, data_size=10000):
        """
        Profile memory usage of indicators.
        
        Args:
            indicator_names: List of indicator names to profile (all if None)
            data_size: Size of the data to use for profiling
            
        Returns:
            Dictionary with memory usage results
        """
        if indicator_names is None:
            indicator_names = list(self.indicators.keys())
        
        logger.info(f"Profiling memory usage for {len(indicator_names)} indicators with {data_size} rows")
        
        data = generate_sample_data(size=data_size)
        memory_results = {}
        
        for name in indicator_names:
            indicator = self.indicators[name]
            
            # Clear any cached data
            gc.collect()
            
            logger.info(f"Profiling memory usage for {name}")
            
            # Start tracking memory allocations
            tracemalloc.start()
            
            # Calculate the indicator
            result = indicator.calculate(data)
            
            # Get memory statistics
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Convert to MB for better readability
            current_mb = current / (1024 * 1024)
            peak_mb = peak / (1024 * 1024)
            
            memory_results[name] = {
                "current_memory_mb": current_mb,
                "peak_memory_mb": peak_mb,
                "result_size_mb": result.memory_usage(deep=True).sum() / (1024 * 1024)
            }
            
            logger.info(f"{name}: Current memory: {current_mb:.2f} MB, Peak memory: {peak_mb:.2f} MB, "
                       f"Result size: {memory_results[name]['result_size_mb']:.2f} MB")
        
        # Save results
        self._save_results("memory_usage", memory_results)
        
        return memory_results
    
    def generate_performance_report(self):
        """
        Generate a comprehensive performance report based on the benchmarks.
        """
        logger.info("Generating performance report")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "data_sizes": self.data_sizes,
            "indicators": list(self.indicators.keys()),
            "results": self.results,
            "memory_results": self.memory_results
        }
        
        # Save report to file
        report_file = os.path.join(self.output_dir, "performance_report.json")
        import json
        with open(report_file, "w") as f:
            json.dump(report, f, default=str, indent=2)
        
        # Generate performance plots
        if self.results:
            self._generate_performance_plots()
        
        logger.info(f"Performance report saved to {report_file}")
        return report
    
    def _save_results(self, name, results):
        """Save benchmark results to a file."""
        import json
        result_file = os.path.join(self.output_dir, f"{name}_results.json")
        with open(result_file, "w") as f:
            json.dump(results, f, default=str, indent=2)
        logger.info(f"Results saved to {result_file}")
    
    def _generate_performance_plots(self):
        """Generate plots visualizing the benchmark results."""
        # Plot calculation speed
        if self.results:
            plt.figure(figsize=(12, 8))
            for indicator, sizes in self.results.items():
                if isinstance(sizes, dict):  # Make sure it's the expected format
                    x = list(sizes.keys())
                    y = list(sizes.values())
                    plt.plot(x, y, marker='o', label=indicator)
            
            plt.title("Indicator Calculation Speed")
            plt.xlabel("Data Size (rows)")
            plt.ylabel("Execution Time (seconds)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            plot_file = os.path.join(self.output_dir, "calculation_speed_plot.png")
            plt.savefig(plot_file)
            logger.info(f"Performance plot saved to {plot_file}")
        
        # Plot memory usage if available
        if self.memory_results:
            plt.figure(figsize=(12, 8))
            indicators = list(self.memory_results.keys())
            peak_memory = [self.memory_results[i]["peak_memory_mb"] for i in indicators]
            result_size = [self.memory_results[i]["result_size_mb"] for i in indicators]
            
            x = range(len(indicators))
            width = 0.35
            
            plt.bar(x, peak_memory, width, label='Peak Memory Usage')
            plt.bar([i + width for i in x], result_size, width, label='Result Size')
            
            plt.title("Indicator Memory Usage")
            plt.xlabel("Indicator")
            plt.ylabel("Memory (MB)")
            plt.xticks([i + width/2 for i in x], indicators, rotation=45)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            plot_file = os.path.join(self.output_dir, "memory_usage_plot.png")
            plt.savefig(plot_file)
            logger.info(f"Memory usage plot saved to {plot_file}")


def run_comprehensive_benchmarks():
    """Run a comprehensive set of benchmarks."""
    benchmark = IndicatorPerformanceBenchmark(
        data_sizes=[1000, 5000, 10000, 20000],
        repeat=3
    )
    
    # Choose a subset of indicators for demonstration
    indicator_subset = [
        "TEMA", "Hull", "KAMA", "AO", "DeMarker", "RVI", "VZO", "VolDelta"
    ]
    
    # Benchmark calculation speed
    benchmark.benchmark_calculation_speed(indicator_subset)
    
    # Profile indicator calculations
    benchmark.profile_indicator_calculations(indicator_subset, data_size=10000)
    
    # Benchmark with caching if available
    benchmark.benchmark_with_caching(indicator_subset, data_size=10000)
    
    # Benchmark multi-timeframe analysis
    result, execution_time = benchmark.benchmark_multi_timeframe_analysis(data_size=10000)
    logger.info(f"Multi-timeframe analysis took {execution_time:.6f} seconds")
    
    # Profile memory usage
    benchmark.memory_results = benchmark.profile_memory_usage(indicator_subset, data_size=10000)
    
    # Generate performance report
    benchmark.generate_performance_report()
    
    return benchmark


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run benchmarks
    benchmark = run_comprehensive_benchmarks()
