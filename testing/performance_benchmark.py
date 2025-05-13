"""
Performance and Accuracy Testing Framework for Phase 4

This module provides comprehensive benchmark tests for indicators
against known trading platforms and measures performance to optimize
slow indicators.
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
import json
import os
import concurrent.futures
from functools import partial
import matplotlib.pyplot as plt
import psutil
import gc
from datetime import datetime
import requests
from pathlib import Path
import sys
import csv

# Add project root to path to ensure imports work
sys.path.append(str(Path(__file__).parent.parent))

from feature_store_service.indicators import (
    moving_averages, 
    oscillators, 
    volatility, 
    volume,
    indicator_registry
)

logger = logging.getLogger(__name__)

class BenchmarkResult:
    """Stores the result of a performance benchmark"""
    
    def __init__(self, 
                 indicator_name: str,
                 execution_time_ms: float,
                 memory_usage_mb: float = None,
                 accuracy_score: float = None,
                 reference_platform: str = None,
                 parameters: Dict[str, Any] = None,
                 data_size: int = None):
        """
        Initialize benchmark result
        
        Args:
            indicator_name: Name of the indicator
            execution_time_ms: Time taken to execute in milliseconds
            memory_usage_mb: Memory used in MB
            accuracy_score: Accuracy compared to reference implementation (1.0 = perfect match)
            reference_platform: Name of reference platform used for comparison
            parameters: Parameters used for the indicator
            data_size: Size of the dataset used
        """
        self.indicator_name = indicator_name
        self.execution_time_ms = execution_time_ms
        self.memory_usage_mb = memory_usage_mb
        self.accuracy_score = accuracy_score
        self.reference_platform = reference_platform
        self.parameters = parameters or {}
        self.data_size = data_size
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark result to dictionary"""
        return {
            "indicator_name": self.indicator_name,
            "execution_time_ms": self.execution_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "accuracy_score": self.accuracy_score,
            "reference_platform": self.reference_platform,
            "parameters": self.parameters,
            "data_size": self.data_size,
            "timestamp": self.timestamp.isoformat()
        }


class PerformanceBenchmark:
    """
    Benchmark indicators for performance and accuracy
    """
    
    def __init__(self, save_path: str = None):
        """
        Initialize performance benchmark
        
        Args:
            save_path: Path to save benchmark results
        """
        self.save_path = save_path or os.path.join("testing", "benchmark_results")
        self.registry = indicator_registry.IndicatorRegistry()
        self.results = []
        self.reference_data = {}
        
        # Ensure save path exists
        os.makedirs(self.save_path, exist_ok=True)
    
    def _generate_test_data(self, size: int = 10000) -> pd.DataFrame:
        """
        Generate test data for benchmarking
        
        Args:
            size: Number of data points
            
        Returns:
            DataFrame with OHLCV data
        """
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=size, freq='1h')
        
        # Base price with trend and cycles
        base_price = np.linspace(100, 200, size)
        cycles = 20 * np.sin(np.linspace(0, 10 * np.pi, size))
        noise = np.random.normal(0, 5, size)
        prices = base_price + cycles + noise
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + np.random.uniform(0, 3, size),
            'low': prices - np.random.uniform(0, 3, size),
            'close': prices + np.random.normal(0, 1, size),
            'volume': np.random.randint(1000, 10000, size)
        })
        
        # Set timestamp as index
        data.set_index('timestamp', inplace=True)
        return data
    
    def measure_execution_time(self, 
                             indicator_fn: Callable, 
                             data: pd.DataFrame,
                             params: Dict[str, Any] = None,
                             repeats: int = 5) -> float:
        """
        Measure execution time of an indicator function
        
        Args:
            indicator_fn: Indicator function
            data: Input data
            params: Parameters for the indicator function
            repeats: Number of times to repeat measurement
            
        Returns:
            Average execution time in milliseconds
        """
        params = params or {}
        execution_times = []
        
        for _ in range(repeats):
            start_time = time.time()
            _ = indicator_fn(data, **params)
            end_time = time.time()
            execution_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return sum(execution_times) / len(execution_times)
    
    def measure_memory_usage(self, 
                           indicator_fn: Callable, 
                           data: pd.DataFrame,
                           params: Dict[str, Any] = None) -> float:
        """
        Measure memory usage of an indicator function
        
        Args:
            indicator_fn: Indicator function
            data: Input data
            params: Parameters for the indicator function
            
        Returns:
            Memory usage in MB
        """
        params = params or {}
        
        # Force garbage collection
        gc.collect()
        
        # Get baseline memory usage
        process = psutil.Process(os.getpid())
        baseline = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        
        # Run indicator
        result = indicator_fn(data, **params)
        
        # Get peak memory usage
        peak = process.memory_info().rss / (1024 * 1024)
        
        # Calculate difference
        memory_used = peak - baseline
        
        # Keep reference to result to prevent premature garbage collection
        result_size = sys.getsizeof(result)
        if isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):
            result_size = result.memory_usage(deep=True).sum() / (1024 * 1024)
            
        return memory_used
    
    def compare_with_reference(self,
                             indicator_fn: Callable,
                             reference_values: pd.DataFrame or pd.Series,
                             data: pd.DataFrame,
                             params: Dict[str, Any] = None) -> float:
        """
        Compare indicator results with reference implementation
        
        Args:
            indicator_fn: Indicator function
            reference_values: Expected output from reference implementation
            data: Input data
            params: Parameters for the indicator function
            
        Returns:
            Accuracy score (1.0 = perfect match)
        """
        params = params or {}
        
        # Calculate indicator
        result = indicator_fn(data, **params)
        
        # Ensure reference and result have same shape
        if isinstance(result, pd.DataFrame) and isinstance(reference_values, pd.DataFrame):
            # Align by index
            result, reference_values = result.align(reference_values, join='inner')
        elif isinstance(result, pd.Series) and isinstance(reference_values, pd.Series):
            # Align by index
            result, reference_values = result.align(reference_values, join='inner')
        else:
            # Try to convert to similar types
            if isinstance(result, pd.DataFrame) and isinstance(reference_values, pd.Series):
                if result.shape[1] == 1:
                    result = result.iloc[:, 0]
            elif isinstance(result, pd.Series) and isinstance(reference_values, pd.DataFrame):
                if reference_values.shape[1] == 1:
                    reference_values = reference_values.iloc[:, 0]
        
        # Calculate difference
        if isinstance(result, pd.DataFrame) and isinstance(reference_values, pd.DataFrame):
            # Normalize values to 0-1 range for each column
            normalized_result = (result - result.min()) / (result.max() - result.min())
            normalized_ref = (reference_values - reference_values.min()) / (reference_values.max() - reference_values.min())
            
            # Calculate mean absolute difference
            diff = (normalized_result - normalized_ref).abs().mean().mean()
            accuracy = 1.0 - min(diff, 1.0)
        else:
            # Normalize values to 0-1 range
            result_min, result_max = result.min(), result.max()
            ref_min, ref_max = reference_values.min(), reference_values.max()
            
            if result_max > result_min and ref_max > ref_min:
                normalized_result = (result - result_min) / (result_max - result_min)
                normalized_ref = (reference_values - ref_min) / (ref_max - ref_min)
                
                # Calculate mean absolute difference
                diff = (normalized_result - normalized_ref).abs().mean()
                accuracy = 1.0 - min(diff, 1.0)
            else:
                # Can't normalize, use correlation instead
                correlation = result.corr(reference_values)
                accuracy = abs(correlation)
        
        return accuracy
    
    def load_reference_data(self, platform: str, indicator: str) -> pd.DataFrame:
        """
        Load reference data for the specified platform and indicator
        
        Args:
            platform: Name of the platform
            indicator: Name of the indicator
            
        Returns:
            DataFrame with reference data
        """
        # Check if already loaded
        key = f"{platform}_{indicator}"
        if key in self.reference_data:
            return self.reference_data[key]
        
        # Path to reference data
        ref_path = os.path.join(self.save_path, "reference_data", platform, f"{indicator}.csv")
        
        try:
            # Load reference data if exists
            if os.path.exists(ref_path):
                data = pd.read_csv(ref_path, index_col=0, parse_dates=True)
                self.reference_data[key] = data
                return data
        except Exception as e:
            logger.error(f"Failed to load reference data: {str(e)}")
        
        return None
    
    def benchmark_indicator(self,
                         indicator_name: str,
                         params: Dict[str, Any] = None,
                         data_size: int = 10000,
                         reference_platform: str = None) -> BenchmarkResult:
        """
        Benchmark an indicator
        
        Args:
            indicator_name: Name of the indicator in the registry
            params: Parameters for the indicator
            data_size: Size of the test dataset
            reference_platform: Name of reference platform for accuracy comparison
            
        Returns:
            BenchmarkResult
        """
        params = params or {}
        
        # Get indicator function
        indicator_info = self.registry.get_indicator(indicator_name)
        if not indicator_info:
            logger.error(f"Indicator {indicator_name} not found in registry")
            return None
            
        indicator_fn = indicator_info["function"]
        
        # Generate test data
        data = self._generate_test_data(data_size)
        
        # Measure execution time
        execution_time = self.measure_execution_time(indicator_fn, data, params)
        
        # Measure memory usage
        memory_usage = self.measure_memory_usage(indicator_fn, data, params)
        
        # Compare with reference if available
        accuracy = None
        if reference_platform:
            reference_data = self.load_reference_data(reference_platform, indicator_name)
            if reference_data is not None:
                accuracy = self.compare_with_reference(indicator_fn, reference_data, data, params)
        
        # Create benchmark result
        result = BenchmarkResult(
            indicator_name=indicator_name,
            execution_time_ms=execution_time,
            memory_usage_mb=memory_usage,
            accuracy_score=accuracy,
            reference_platform=reference_platform,
            parameters=params,
            data_size=data_size
        )
        
        # Save result
        self.results.append(result)
        
        return result
    
    def benchmark_all(self,
                    params_dict: Dict[str, Dict[str, Any]] = None,
                    data_size: int = 10000,
                    reference_platform: str = None,
                    parallel: bool = True) -> List[BenchmarkResult]:
        """
        Benchmark all registered indicators
        
        Args:
            params_dict: Dictionary of parameters for each indicator
            data_size: Size of the test dataset
            reference_platform: Name of reference platform for accuracy comparison
            parallel: Whether to run benchmarks in parallel
            
        Returns:
            List of BenchmarkResults
        """
        params_dict = params_dict or {}
        
        # Get all indicators
        indicators = self.registry.get_all_indicators()
        
        results = []
        
        if parallel:
            # Create partial function with fixed parameters
            def benchmark_wrapper(indicator):
    """
    Benchmark wrapper.
    
    Args:
        indicator: Description of indicator
    
    """

                try:
                    params = params_dict.get(indicator, {})
                    return self.benchmark_indicator(
                        indicator_name=indicator,
                        params=params,
                        data_size=data_size,
                        reference_platform=reference_platform
                    )
                except Exception as e:
                    logger.error(f"Failed to benchmark {indicator}: {str(e)}")
                    return None
            
            # Run in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(benchmark_wrapper, indicator) for indicator in indicators]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)
        else:
            # Run sequentially
            for indicator in indicators:
                try:
                    params = params_dict.get(indicator, {})
                    result = self.benchmark_indicator(
                        indicator_name=indicator,
                        params=params,
                        data_size=data_size,
                        reference_platform=reference_platform
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Failed to benchmark {indicator}: {str(e)}")
        
        return results
    
    def identify_slow_indicators(self, threshold_ms: int = 100) -> List[BenchmarkResult]:
        """
        Identify slow indicators that could be optimized
        
        Args:
            threshold_ms: Threshold in milliseconds to consider an indicator slow
            
        Returns:
            List of BenchmarkResults for slow indicators
        """
        slow_indicators = [r for r in self.results if r.execution_time_ms > threshold_ms]
        slow_indicators.sort(key=lambda x: x.execution_time_ms, reverse=True)
        return slow_indicators
    
    def save_results(self, filename: str = None):
        """
        Save benchmark results to JSON file
        
        Args:
            filename: Name of the file to save to
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
            
        filepath = os.path.join(self.save_path, filename)
        
        # Convert results to dictionary
        results_dict = [r.to_dict() for r in self.results]
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
            
        logger.info(f"Saved {len(results_dict)} benchmark results to {filepath}")
    
    def load_results(self, filepath: str) -> int:
        """
        Load benchmark results from file
        
        Args:
            filepath: Path to the file to load from
            
        Returns:
            Number of results loaded
        """
        try:
            with open(filepath, 'r') as f:
                results_dict = json.load(f)
                
            # Convert dictionaries to BenchmarkResult objects
            for result_dict in results_dict:
                # Convert timestamp string to datetime
                timestamp = datetime.fromisoformat(result_dict["timestamp"])
                
                result = BenchmarkResult(
                    indicator_name=result_dict["indicator_name"],
                    execution_time_ms=result_dict["execution_time_ms"],
                    memory_usage_mb=result_dict["memory_usage_mb"],
                    accuracy_score=result_dict["accuracy_score"],
                    reference_platform=result_dict["reference_platform"],
                    parameters=result_dict["parameters"],
                    data_size=result_dict["data_size"]
                )
                result.timestamp = timestamp
                self.results.append(result)
                
            logger.info(f"Loaded {len(results_dict)} benchmark results from {filepath}")
            return len(results_dict)
        except Exception as e:
            logger.error(f"Failed to load benchmark results: {str(e)}")
            return 0
    
    def plot_execution_times(self, top_n: int = None):
        """
        Plot execution times of indicators
        
        Args:
            top_n: Number of top indicators to show (by execution time)
        """
        if not self.results:
            logger.warning("No benchmark results to plot")
            return
            
        # Sort by execution time
        sorted_results = sorted(self.results, key=lambda x: x.execution_time_ms, reverse=True)
        
        # Limit to top_n if specified
        if top_n:
            sorted_results = sorted_results[:top_n]
            
        # Create plot
        indicators = [r.indicator_name for r in sorted_results]
        times = [r.execution_time_ms for r in sorted_results]
        
        plt.figure(figsize=(12, 8))
        plt.barh(indicators, times, color='skyblue')
        plt.xlabel('Execution Time (ms)')
        plt.ylabel('Indicator')
        plt.title('Indicator Execution Times')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_path, "execution_times.png")
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Saved execution time plot to {plot_path}")
    
    def plot_accuracy_comparison(self, reference_platform: str = None):
        """
        Plot accuracy comparison with reference platform
        
        Args:
            reference_platform: Name of reference platform
        """
        # Filter by reference platform if specified
        if reference_platform:
            results = [r for r in self.results if r.reference_platform == reference_platform]
        else:
            results = [r for r in self.results if r.accuracy_score is not None]
            
        if not results:
            logger.warning("No benchmark results with accuracy scores to plot")
            return
            
        # Sort by accuracy
        sorted_results = sorted(results, key=lambda x: x.accuracy_score)
        
        # Create plot
        indicators = [r.indicator_name for r in sorted_results]
        accuracy = [r.accuracy_score for r in sorted_results]
        
        plt.figure(figsize=(12, 8))
        plt.barh(indicators, accuracy, color='lightgreen')
        plt.xlabel('Accuracy Score')
        plt.ylabel('Indicator')
        plt.title(f'Indicator Accuracy Comparison with {reference_platform or "Reference"}')
        plt.xlim(0, 1)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_path, "accuracy_comparison.png")
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Saved accuracy comparison plot to {plot_path}")
    
    def generate_report(self, filename: str = None):
        """
        Generate a detailed benchmark report
        
        Args:
            filename: Name of the file to save the report to
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_report_{timestamp}.html"
            
        filepath = os.path.join(self.save_path, filename)
        
        # Sort results by different criteria
        by_time = sorted(self.results, key=lambda x: x.execution_time_ms, reverse=True)
        by_memory = sorted(self.results, key=lambda x: x.memory_usage_mb or 0, reverse=True)
        by_accuracy = sorted(
            [r for r in self.results if r.accuracy_score is not None], 
            key=lambda x: x.accuracy_score
        )
        
        # Create HTML report
        report = []
        report.append("<!DOCTYPE html>")
        report.append("<html lang='en'>")
        report.append("<head>")
        report.append("<meta charset='UTF-8'>")
        report.append("<meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        report.append("<title>Indicator Benchmark Report</title>")
        report.append("<style>")
        report.append("body { font-family: Arial, sans-serif; margin: 40px; }")
        report.append("table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }")
        report.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        report.append("th { background-color: #f2f2f2; }")
        report.append("tr:hover { background-color: #f5f5f5; }")
        report.append("h1, h2 { color: #333; }")
        report.append(".summary { background-color: #eef; padding: 15px; border-radius: 5px; }")
        report.append("</style>")
        report.append("</head>")
        report.append("<body>")
        report.append("<h1>Indicator Benchmark Report</h1>")
        
        # Summary
        report.append("<div class='summary'>")
        report.append(f"<p><strong>Total Indicators:</strong> {len(self.results)}</p>")
        report.append(f"<p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        platforms = set([r.reference_platform for r in self.results if r.reference_platform])
        report.append(f"<p><strong>Reference Platforms:</strong> {', '.join(platforms) if platforms else 'None'}</p>")
        
        # Performance overview
        if by_time:
            avg_time = sum(r.execution_time_ms for r in self.results) / len(self.results)
            report.append(f"<p><strong>Average Execution Time:</strong> {avg_time:.2f} ms</p>")
            report.append(f"<p><strong>Slowest Indicator:</strong> {by_time[0].indicator_name} ({by_time[0].execution_time_ms:.2f} ms)</p>")
            report.append(f"<p><strong>Fastest Indicator:</strong> {by_time[-1].indicator_name} ({by_time[-1].execution_time_ms:.2f} ms)</p>")
        
        # Memory overview
        memory_results = [r for r in self.results if r.memory_usage_mb is not None]
        if memory_results:
            avg_memory = sum(r.memory_usage_mb for r in memory_results) / len(memory_results)
            report.append(f"<p><strong>Average Memory Usage:</strong> {avg_memory:.2f} MB</p>")
            report.append(f"<p><strong>Highest Memory Usage:</strong> {by_memory[0].indicator_name} ({by_memory[0].memory_usage_mb:.2f} MB)</p>")
            report.append(f"<p><strong>Lowest Memory Usage:</strong> {by_memory[-1].indicator_name} ({by_memory[-1].memory_usage_mb:.2f} MB)</p>")
        
        # Accuracy overview
        accuracy_results = [r for r in self.results if r.accuracy_score is not None]
        if accuracy_results:
            avg_accuracy = sum(r.accuracy_score for r in accuracy_results) / len(accuracy_results)
            report.append(f"<p><strong>Average Accuracy Score:</strong> {avg_accuracy:.4f}</p>")
            if by_accuracy:
                report.append(f"<p><strong>Lowest Accuracy:</strong> {by_accuracy[0].indicator_name} ({by_accuracy[0].accuracy_score:.4f})</p>")
                report.append(f"<p><strong>Highest Accuracy:</strong> {by_accuracy[-1].indicator_name} ({by_accuracy[-1].accuracy_score:.4f})</p>")
        
        report.append("</div>")
        
        # Execution Time Table
        report.append("<h2>Indicators by Execution Time</h2>")
        report.append("<table>")
        report.append("<tr><th>Indicator</th><th>Execution Time (ms)</th><th>Parameters</th></tr>")
        for r in by_time[:20]:  # Show top 20
            params_str = ", ".join([f"{k}={v}" for k, v in r.parameters.items()]) if r.parameters else "-"
            report.append(f"<tr><td>{r.indicator_name}</td><td>{r.execution_time_ms:.2f}</td><td>{params_str}</td></tr>")
        report.append("</table>")
        
        # Memory Usage Table
        if memory_results:
            report.append("<h2>Indicators by Memory Usage</h2>")
            report.append("<table>")
            report.append("<tr><th>Indicator</th><th>Memory Usage (MB)</th><th>Parameters</th></tr>")
            for r in by_memory[:20]:  # Show top 20
                if r.memory_usage_mb is not None:
                    params_str = ", ".join([f"{k}={v}" for k, v in r.parameters.items()]) if r.parameters else "-"
                    report.append(f"<tr><td>{r.indicator_name}</td><td>{r.memory_usage_mb:.2f}</td><td>{params_str}</td></tr>")
            report.append("</table>")
        
        # Accuracy Table
        if accuracy_results:
            report.append("<h2>Indicators by Accuracy</h2>")
            report.append("<table>")
            report.append("<tr><th>Indicator</th><th>Accuracy Score</th><th>Reference Platform</th></tr>")
            for r in by_accuracy:
                report.append(f"<tr><td>{r.indicator_name}</td><td>{r.accuracy_score:.4f}</td><td>{r.reference_platform or '-'}</td></tr>")
            report.append("</table>")
        
        # Optimization Recommendations
        report.append("<h2>Optimization Recommendations</h2>")
        slow_indicators = self.identify_slow_indicators()
        if slow_indicators:
            report.append("<p>The following indicators could benefit from optimization:</p>")
            report.append("<ul>")
            for r in slow_indicators[:10]:  # Top 10 slowest
                report.append(f"<li><strong>{r.indicator_name}</strong> - {r.execution_time_ms:.2f} ms</li>")
            report.append("</ul>")
        else:
            report.append("<p>All indicators are performing within acceptable time limits.</p>")
        
        report.append("</body>")
        report.append("</html>")
        
        # Save report
        with open(filepath, 'w') as f:
            f.write("\n".join(report))
        
        logger.info(f"Generated benchmark report at {filepath}")
        return filepath


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    benchmark = PerformanceBenchmark()
    
    # Register some indicators for testing
    registry = indicator_registry.IndicatorRegistry()
    registry.register_common_indicators()
    
    # Run benchmark
    results = benchmark.benchmark_all(data_size=5000)
    
    # Generate report
    benchmark.generate_report()
    
    # Plot results
    benchmark.plot_execution_times(top_n=20)
""""""
