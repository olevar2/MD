"""
Indicator Optimizer Module

This module uses the performance benchmark tool to identify slow indicators
and provides optimized implementations for common performance bottlenecks.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import pandas as pd
import numpy as np
import time
import functools
import inspect
import cProfile
import pstats
import io
from pathlib import Path
import json
import os
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numba
from numba import jit, prange, float64, int64

# Add necessary paths to system path
REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(REPO_ROOT))

from testing.performance_benchmark import PerformanceBenchmark, BenchmarkResult
from feature_store_service.indicators import indicator_registry

logger = logging.getLogger(__name__)

class IndicatorOptimizer:
    """
    Class for optimizing indicator performance using various techniques
    """
    
    def __init__(self, benchmark_results_path: str = None):
        """
        Initialize the optimizer
        
        Args:
            benchmark_results_path: Path to benchmark results (optional)
        """
        self.benchmark = PerformanceBenchmark()
        self.registry = indicator_registry.IndicatorRegistry()
        self.registry.register_common_indicators()
        self.optimized_indicators = {}
        
        # Load benchmark results if provided
        if benchmark_results_path and os.path.exists(benchmark_results_path):
            self.benchmark.load_results(benchmark_results_path)
    
    def run_benchmarks(self, data_size: int = 10000) -> List[BenchmarkResult]:
        """
        Run benchmarks on all indicators to identify slow ones
        
        Args:
            data_size: Size of the test data set
            
        Returns:
            List of benchmark results
        """
        # Run benchmarks
        results = self.benchmark.benchmark_all(data_size=data_size)
        
        # Sort by execution time
        results.sort(key=lambda r: r.execution_time_ms, reverse=True)
        
        # Log results
        logger.info(f"Benchmarked {len(results)} indicators")
        for i, result in enumerate(results[:10]):
            logger.info(f"#{i+1} Slow indicator: {result.indicator_name} - {result.execution_time_ms:.2f} ms")
        
        return results
    
    def identify_slow_indicators(self, threshold_ms: int = 100) -> List[str]:
        """
        Identify indicators that need optimization
        
        Args:
            threshold_ms: Threshold in milliseconds to consider an indicator slow
            
        Returns:
            List of indicator names
        """
        # Get slow indicators from benchmark results
        slow_results = self.benchmark.identify_slow_indicators(threshold_ms)
        slow_indicators = [result.indicator_name for result in slow_results]
        
        # If no benchmark results, run benchmarks
        if not slow_indicators:
            results = self.run_benchmarks()
            slow_results = [r for r in results if r.execution_time_ms > threshold_ms]
            slow_indicators = [result.indicator_name for result in slow_results]
        
        return slow_indicators
    
    def profile_indicator(self, indicator_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Profile an indicator to identify bottlenecks
        
        Args:
            indicator_name: Name of the indicator to profile
            params: Parameters for the indicator
            
        Returns:
            Dictionary with profiling information
        """
        params = params or {}
        
        # Get indicator function
        indicator_info = self.registry.get_indicator(indicator_name)
        if not indicator_info:
            logger.error(f"Indicator {indicator_name} not found in registry")
            return {}
        
        indicator_fn = indicator_info["function"]
        
        # Generate test data
        data = self.benchmark._generate_test_data(5000)
        
        # Profile the function
        pr = cProfile.Profile()
        pr.enable()
        result = indicator_fn(data, **params)
        pr.disable()
        
        # Convert profiling results to string
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 time-consuming functions
        
        # Extract function calls and times
        lines = s.getvalue().split('\n')
        
        # Parse profiling output
        profile_data = {
            "indicator": indicator_name,
            "parameters": params,
            "top_calls": []
        }
        
        parsing = False
        for line in lines:
            if line.strip().startswith("ncalls"):
                parsing = True
                continue
                
            if parsing and line.strip():
                parts = line.strip().split()
                if len(parts) >= 6:
                    try:
                        profile_data["top_calls"].append({
                            "ncalls": parts[0],
                            "tottime": float(parts[1]),
                            "percall": float(parts[2]),
                            "cumtime": float(parts[3]),
                            "percall_cum": float(parts[4]),
                            "function": ' '.join(parts[5:])
                        })
                    except ValueError:
                        continue
        
        return profile_data
    
    def optimize_with_numba(self, indicator_fn: Callable) -> Callable:
        """
        Optimize a function using Numba JIT compilation
        
        Args:
            indicator_fn: Function to optimize
            
        Returns:
            Optimized function
        """
        # Get function source code and signature
        source = inspect.getsource(indicator_fn)
        sig = inspect.signature(indicator_fn)
        
        # Check if function is suitable for Numba optimization
        is_suitable = True
        try:
            # Attempt to JIT compile the function
            optimized_fn = numba.jit(nopython=True)(indicator_fn)
            # Test with dummy data
            test_data = pd.DataFrame({
                'open': np.random.random(100),
                'high': np.random.random(100),
                'low': np.random.random(100),
                'close': np.random.random(100),
                'volume': np.random.random(100)
            })
            _ = optimized_fn(test_data.values)
        except Exception as e:
            logger.warning(f"Function not suitable for Numba optimization: {str(e)}")
            is_suitable = False
        
        if is_suitable:
            # Return the optimized function
            return optimized_fn
        else:
            # Return original function if optimization failed
            return indicator_fn
    
    def _create_numba_optimized_moving_average(self):
        """Create Numba-optimized implementation of moving average indicators"""
        
        @jit(nopython=True)
        def optimized_sma(close_prices, period):
            """Numba-optimized Simple Moving Average"""
            n = len(close_prices)
            result = np.empty(n)
            result[:] = np.nan
            
            for i in range(period - 1, n):
                result[i] = np.mean(close_prices[i - period + 1:i + 1])
                
            return result
        
        @jit(nopython=True)
        def optimized_ema(close_prices, period):
            """Numba-optimized Exponential Moving Average"""
            n = len(close_prices)
            result = np.empty(n)
            result[:] = np.nan
            
            # Initialize with SMA
            result[period - 1] = np.mean(close_prices[:period])
            
            # EMA multiplier
            multiplier = 2 / (period + 1)
            
            # Calculate EMA
            for i in range(period, n):
                result[i] = close_prices[i] * multiplier + result[i - 1] * (1 - multiplier)
                
            return result
        
        # Wrapper function for SMA that handles pandas input
        def optimized_sma_wrapper(data, period=20, price_column='close'):
    """
    Optimized sma wrapper.
    
    Args:
        data: Description of data
        period: Description of period
        price_column: Description of price_column
    
    """

            if isinstance(data, pd.DataFrame):
                prices = data[price_column].values
            else:
                prices = data
                
            result = optimized_sma(prices, period)
            
            # Return as pandas Series with original index if input was DataFrame
            if isinstance(data, pd.DataFrame):
                return pd.Series(result, index=data.index)
            return result
        
        # Wrapper function for EMA that handles pandas input
        def optimized_ema_wrapper(data, period=20, price_column='close'):
    """
    Optimized ema wrapper.
    
    Args:
        data: Description of data
        period: Description of period
        price_column: Description of price_column
    
    """

            if isinstance(data, pd.DataFrame):
                prices = data[price_column].values
            else:
                prices = data
                
            result = optimized_ema(prices, period)
            
            # Return as pandas Series with original index if input was DataFrame
            if isinstance(data, pd.DataFrame):
                return pd.Series(result, index=data.index)
            return result
        
        # Register optimized functions
        self.optimized_indicators['simple_moving_average'] = optimized_sma_wrapper
        self.optimized_indicators['exponential_moving_average'] = optimized_ema_wrapper
        
        logger.info("Created Numba-optimized implementations for moving averages")
    
    def _create_numba_optimized_rsi(self):
        """Create Numba-optimized implementation of RSI"""
        
        @jit(nopython=True)
        def optimized_rsi(close_prices, period):
            """Numba-optimized Relative Strength Index"""
            n = len(close_prices)
            result = np.empty(n)
            result[:] = np.nan
            
            # Calculate price changes
            changes = np.zeros(n)
            for i in range(1, n):
                changes[i] = close_prices[i] - close_prices[i - 1]
            
            # Separate gains and losses
            gains = np.zeros(n)
            losses = np.zeros(n)
            
            for i in range(1, n):
                if changes[i] > 0:
                    gains[i] = changes[i]
                elif changes[i] < 0:
                    losses[i] = -changes[i]
            
            # Calculate average gains and losses
            avg_gains = np.zeros(n)
            avg_losses = np.zeros(n)
            
            # Initial averages
            avg_gains[period] = np.sum(gains[1:period+1]) / period
            avg_losses[period] = np.sum(losses[1:period+1]) / period
            
            # Calculate smoothed averages
            for i in range(period + 1, n):
                avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i]) / period
                avg_losses[i] = (avg_losses[i - 1] * (period - 1) + losses[i]) / period
            
            # Calculate RS and RSI
            for i in range(period, n):
                if avg_losses[i] == 0:
                    result[i] = 100
                else:
                    rs = avg_gains[i] / avg_losses[i]
                    result[i] = 100 - (100 / (1 + rs))
            
            return result
        
        # Wrapper function for RSI that handles pandas input
        def optimized_rsi_wrapper(data, period=14, price_column='close'):
    """
    Optimized rsi wrapper.
    
    Args:
        data: Description of data
        period: Description of period
        price_column: Description of price_column
    
    """

            if isinstance(data, pd.DataFrame):
                prices = data[price_column].values
            else:
                prices = data
                
            result = optimized_rsi(prices, period)
            
            # Return as pandas Series with original index if input was DataFrame
            if isinstance(data, pd.DataFrame):
                return pd.Series(result, index=data.index)
            return result
        
        # Register optimized function
        self.optimized_indicators['relative_strength_index'] = optimized_rsi_wrapper
        
        logger.info("Created Numba-optimized implementation for RSI")
    
    def _create_numba_optimized_bollinger_bands(self):
        """Create Numba-optimized implementation of Bollinger Bands"""
        
        @jit(nopython=True)
        def optimized_bollinger(close_prices, period, deviations):
            """Numba-optimized Bollinger Bands"""
            n = len(close_prices)
            middle = np.empty(n)
            upper = np.empty(n)
            lower = np.empty(n)
            
            middle[:] = np.nan
            upper[:] = np.nan
            lower[:] = np.nan
            
            # Calculate moving average and standard deviation
            for i in range(period - 1, n):
                window = close_prices[i - period + 1:i + 1]
                middle[i] = np.mean(window)
                std = np.std(window)
                upper[i] = middle[i] + (std * deviations)
                lower[i] = middle[i] - (std * deviations)
            
            return middle, upper, lower
        
        # Wrapper function for Bollinger Bands that handles pandas input
        def optimized_bollinger_wrapper(data, period=20, deviations=2.0, price_column='close'):
    """
    Optimized bollinger wrapper.
    
    Args:
        data: Description of data
        period: Description of period
        deviations: Description of deviations
        price_column: Description of price_column
    
    """

            if isinstance(data, pd.DataFrame):
                prices = data[price_column].values
            else:
                prices = data
                
            middle, upper, lower = optimized_bollinger(prices, period, deviations)
            
            # Return as pandas DataFrame with original index if input was DataFrame
            if isinstance(data, pd.DataFrame):
                result = pd.DataFrame({
                    'middle': middle,
                    'upper': upper,
                    'lower': lower
                }, index=data.index)
            else:
                result = pd.DataFrame({
                    'middle': middle,
                    'upper': upper,
                    'lower': lower
                })
            
            return result
        
        # Register optimized function
        self.optimized_indicators['bollinger_bands'] = optimized_bollinger_wrapper
        
        logger.info("Created Numba-optimized implementation for Bollinger Bands")
    
    def create_optimized_implementations(self):
        """Create optimized implementations for common indicators"""
        # Create optimized versions of commonly used indicators
        self._create_numba_optimized_moving_average()
        self._create_numba_optimized_rsi()
        self._create_numba_optimized_bollinger_bands()
        
        # Add more optimized implementations as needed
    
    def benchmark_original_vs_optimized(self, indicator_name: str, 
                                     params: Dict[str, Any] = None,
                                     data_size: int = 10000,
                                     repeats: int = 5) -> Dict[str, Any]:
        """
        Compare performance of original vs optimized indicator implementations
        
        Args:
            indicator_name: Name of the indicator
            params: Parameters for the indicator
            data_size: Size of the test data
            repeats: Number of times to repeat the test
            
        Returns:
            Dictionary with benchmark results
        """
        params = params or {}
        
        # Get original indicator function
        original_info = self.registry.get_indicator(indicator_name)
        if not original_info:
            logger.error(f"Indicator {indicator_name} not found in registry")
            return {}
            
        original_fn = original_info["function"]
        
        # Get optimized implementation if available
        if indicator_name in self.optimized_indicators:
            optimized_fn = self.optimized_indicators[indicator_name]
        else:
            logger.warning(f"No optimized implementation available for {indicator_name}")
            return {}
        
        # Generate test data
        data = self.benchmark._generate_test_data(data_size)
        
        # Benchmark original implementation
        original_times = []
        for _ in range(repeats):
            start_time = time.time()
            _ = original_fn(data, **params)
            end_time = time.time()
            original_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        original_avg = sum(original_times) / len(original_times)
        
        # Benchmark optimized implementation
        optimized_times = []
        for _ in range(repeats):
            start_time = time.time()
            _ = optimized_fn(data, **params)
            end_time = time.time()
            optimized_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        optimized_avg = sum(optimized_times) / len(optimized_times)
        
        # Calculate improvement
        improvement_pct = ((original_avg - optimized_avg) / original_avg) * 100
        
        results = {
            "indicator": indicator_name,
            "parameters": params,
            "data_size": data_size,
            "original_time_ms": original_avg,
            "optimized_time_ms": optimized_avg,
            "improvement_pct": improvement_pct,
            "original_times": original_times,
            "optimized_times": optimized_times
        }
        
        logger.info(f"Benchmark results for {indicator_name}: "
                  f"Original: {original_avg:.2f} ms, "
                  f"Optimized: {optimized_avg:.2f} ms, "
                  f"Improvement: {improvement_pct:.2f}%")
        
        return results
    
    def register_optimized_indicators(self):
        """Register optimized indicator implementations with the registry"""
        for name, fn in self.optimized_indicators.items():
            # Get original indicator info
            original_info = self.registry.get_indicator(name)
            if not original_info:
                logger.warning(f"Cannot register optimized indicator {name}: original not found")
                continue
            
            # Register with same metadata but optimized function
            self.registry.register(
                name, 
                fn,
                description=original_info.get("description", f"Optimized {name}"),
                category=original_info.get("category", "optimized"),
                parameters=original_info.get("parameters", {})
            )
            
            logger.info(f"Registered optimized implementation of {name}")
    
    def optimize_indicator(self, indicator_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize a specific indicator and measure improvement
        
        Args:
            indicator_name: Name of the indicator to optimize
            params: Parameters for the indicator
            
        Returns:
            Dictionary with optimization results
        """
        params = params or {}
        
        # Profile the indicator to identify bottlenecks
        profile_data = self.profile_indicator(indicator_name, params)
        
        # If already optimized, compare performance
        if indicator_name in self.optimized_indicators:
            benchmark_results = self.benchmark_original_vs_optimized(indicator_name, params)
            return {
                "indicator": indicator_name,
                "parameters": params,
                "profile_data": profile_data,
                "benchmark_results": benchmark_results,
                "already_optimized": True
            }
        
        # For indicators without pre-built optimized versions, attempt automatic optimization
        indicator_info = self.registry.get_indicator(indicator_name)
        if not indicator_info:
            logger.error(f"Indicator {indicator_name} not found in registry")
            return {}
            
        original_fn = indicator_info["function"]
        
        # Attempt Numba optimization
        logger.info(f"Attempting automatic optimization of {indicator_name}")
        optimized_fn = self.optimize_with_numba(original_fn)
        
        # Store the optimized function
        self.optimized_indicators[indicator_name] = optimized_fn
        
        # Compare performance
        benchmark_results = self.benchmark_original_vs_optimized(indicator_name, params)
        
        return {
            "indicator": indicator_name,
            "parameters": params,
            "profile_data": profile_data,
            "benchmark_results": benchmark_results,
            "optimization_method": "numba_jit",
            "already_optimized": False
        }
    
    def optimize_all_slow_indicators(self, threshold_ms: int = 100) -> Dict[str, Dict[str, Any]]:
        """
        Optimize all indicators that are slower than the threshold
        
        Args:
            threshold_ms: Threshold in milliseconds to consider an indicator slow
            
        Returns:
            Dictionary mapping indicator names to their optimization results
        """
        # Ensure we have optimized implementations
        self.create_optimized_implementations()
        
        # Identify slow indicators
        slow_indicators = self.identify_slow_indicators(threshold_ms)
        
        # Optimize each slow indicator
        results = {}
        for indicator_name in slow_indicators:
            # Use default parameters
            results[indicator_name] = self.optimize_indicator(indicator_name)
        
        # Register optimized indicators
        self.register_optimized_indicators()
        
        return results
    
    def generate_optimization_report(self, results: Dict[str, Dict[str, Any]], output_path: str = None) -> str:
        """
        Generate an HTML report of optimization results
        
        Args:
            results: Dictionary of optimization results
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        if not output_path:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(REPO_ROOT, "testing", "reports", f"optimization_report_{timestamp}.html")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Sort results by improvement percentage
        sorted_results = sorted(
            [(name, data) for name, data in results.items() if "benchmark_results" in data],
            key=lambda x: x[1]["benchmark_results"].get("improvement_pct", 0),
            reverse=True
        )
        
        # Generate HTML report
        html_lines = []
        html_lines.append("<!DOCTYPE html>")
        html_lines.append("<html lang='en'>")
        html_lines.append("<head>")
        html_lines.append("    <meta charset='UTF-8'>")
        html_lines.append("    <title>Indicator Optimization Report</title>")
        html_lines.append("    <style>")
        html_lines.append("        body { font-family: Arial, sans-serif; margin: 40px; }")
        html_lines.append("        h1, h2 { color: #333; }")
        html_lines.append("        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
        html_lines.append("        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html_lines.append("        th { background-color: #f2f2f2; }")
        html_lines.append("        tr:hover { background-color: #f5f5f5; }")
        html_lines.append("        .positive { color: green; }")
        html_lines.append("        .negative { color: red; }")
        html_lines.append("        .summary { background-color: #eef; padding: 15px; border-radius: 5px; margin-bottom: 30px; }")
        html_lines.append("    </style>")
        html_lines.append("</head>")
        html_lines.append("<body>")
        
        # Header
        html_lines.append("    <h1>Indicator Optimization Report</h1>")
        
        # Summary
        html_lines.append("    <div class='summary'>")
        html_lines.append(f"    <p><strong>Total Indicators Analyzed:</strong> {len(results)}</p>")
        html_lines.append(f"    <p><strong>Date:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Calculate overall improvement
        if sorted_results:
            total_original = sum(r[1]["benchmark_results"].get("original_time_ms", 0) for r in sorted_results)
            total_optimized = sum(r[1]["benchmark_results"].get("optimized_time_ms", 0) for r in sorted_results)
            overall_improvement = ((total_original - total_optimized) / total_original) * 100 if total_original > 0 else 0
            
            html_lines.append(f"    <p><strong>Overall Performance Improvement:</strong> {overall_improvement:.2f}%</p>")
            
            # Best improvement
            best = sorted_results[0] if sorted_results else (None, {})
            if best[0]:
                best_pct = best[1]["benchmark_results"].get("improvement_pct", 0)
                html_lines.append(f"    <p><strong>Best Improvement:</strong> {best[0]} ({best_pct:.2f}%)</p>")
        
        html_lines.append("    </div>")
        
        # Results table
        html_lines.append("    <h2>Optimization Results</h2>")
        html_lines.append("    <table>")
        html_lines.append("        <tr>")
        html_lines.append("            <th>Indicator</th>")
        html_lines.append("            <th>Original Time (ms)</th>")
        html_lines.append("            <th>Optimized Time (ms)</th>")
        html_lines.append("            <th>Improvement</th>")
        html_lines.append("            <th>Method</th>")
        html_lines.append("        </tr>")
        
        for name, data in sorted_results:
            benchmark = data.get("benchmark_results", {})
            original = benchmark.get("original_time_ms", 0)
            optimized = benchmark.get("optimized_time_ms", 0)
            improvement = benchmark.get("improvement_pct", 0)
            method = data.get("optimization_method", "pre-built")
            
            improvement_class = "positive" if improvement > 0 else "negative"
            
            html_lines.append("        <tr>")
            html_lines.append(f"            <td>{name}</td>")
            html_lines.append(f"            <td>{original:.2f}</td>")
            html_lines.append(f"            <td>{optimized:.2f}</td>")
            html_lines.append(f"            <td class='{improvement_class}'>{improvement:.2f}%</td>")
            html_lines.append(f"            <td>{method}</td>")
            html_lines.append("        </tr>")
        
        html_lines.append("    </table>")
        
        # Detailed results for each indicator
        html_lines.append("    <h2>Detailed Results</h2>")
        
        for name, data in sorted_results:
            html_lines.append(f"    <h3>{name}</h3>")
            
            # Parameters
            params = data.get("parameters", {})
            if params:
                html_lines.append("    <p><strong>Parameters:</strong></p>")
                html_lines.append("    <ul>")
                for param, value in params.items():
                    html_lines.append(f"        <li>{param}: {value}</li>")
                html_lines.append("    </ul>")
            
            # Profile data
            profile_data = data.get("profile_data", {})
            top_calls = profile_data.get("top_calls", [])
            
            if top_calls:
                html_lines.append("    <p><strong>Top Time-Consuming Functions:</strong></p>")
                html_lines.append("    <table>")
                html_lines.append("        <tr>")
                html_lines.append("            <th>Function</th>")
                html_lines.append("            <th>Calls</th>")
                html_lines.append("            <th>Cumulative Time (s)</th>")
                html_lines.append("        </tr>")
                
                for call in top_calls[:5]:  # Show top 5
                    html_lines.append("        <tr>")
                    html_lines.append(f"            <td>{call.get('function', '')}</td>")
                    html_lines.append(f"            <td>{call.get('ncalls', '')}</td>")
                    html_lines.append(f"            <td>{call.get('cumtime', 0)}</td>")
                    html_lines.append("        </tr>")
                
                html_lines.append("    </table>")
        
        html_lines.append("</body>")
        html_lines.append("</html>")
        
        # Write HTML report to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(html_lines))
        
        logger.info(f"Generated optimization report at {output_path}")
        return output_path


# Example usage when run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create optimizer
    optimizer = IndicatorOptimizer()
    
    # Create optimized implementations
    optimizer.create_optimized_implementations()
    
    # Run optimization
    results = optimizer.optimize_all_slow_indicators()
    
    # Generate report
    optimizer.generate_optimization_report(results)
""""""
