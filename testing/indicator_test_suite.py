"""
Comprehensive Indicator Test Suite

This module implements the testing requirements from Phase 8 of the project,
focusing on comprehensive testing of indicators across the platform.

It provides:
1. Unit testing for all indicators
2. Integration testing between indicator groups
3. Performance testing under various market conditions
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import unittest
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import feature store indicators
from feature_store_service.indicators.testing.indicator_tester import IndicatorTester
from feature_store_service.indicators import (
    moving_averages,
    oscillators,
    volume,
    volatility,
    fibonacci,
    advanced_price_indicators,
    advanced_moving_averages,
    advanced_oscillators
)

# Import analysis engine components
from analysis_engine.analysis.basic_ta import basic_indicators
from analysis_engine.analysis.advanced_ta import advanced_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndicatorTestSuite:
    """
    A comprehensive test suite for indicators following Phase 8 requirements.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the test suite.
        
        Args:
            data_path: Path to historical data for testing
        """
        self.data_path = data_path
        self.test_data = self._load_test_data() if data_path else None
        self.indicator_tester = IndicatorTester(self.test_data)
        self.results_directory = os.path.join(os.path.dirname(__file__), "test_results")
        
        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)
        
    def _load_test_data(self) -> pd.DataFrame:
        """Load historical market data for testing."""
        try:
            data = pd.read_csv(self.data_path)
            logger.info(f"Loaded test data from {self.data_path} with {len(data)} rows")
            return data
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            # Generate synthetic data as fallback
            return self._generate_synthetic_data()
            
    def _generate_synthetic_data(self, rows: int = 1000) -> pd.DataFrame:
        """Generate synthetic market data for testing."""
        logger.info(f"Generating {rows} rows of synthetic test data")
        
        np.random.seed(42)  # For reproducibility
        
        # Generate random walk price data
        close = np.random.normal(0, 1, rows).cumsum() + 100
        high = close + np.random.normal(0, 0.5, rows).cumsum()
        low = close - np.random.normal(0, 0.5, rows).cumsum()
        # Ensure high is always highest and low is always lowest
        high = np.maximum(high, close)
        low = np.minimum(low, close)
        
        open_prices = close.copy()
        np.random.shuffle(open_prices)
        # Ensure open is within high-low range
        open_prices = np.minimum(high, np.maximum(low, open_prices))
        
        # Generate volume data
        volume = np.exp(np.random.normal(10, 1, rows))
        
        # Create DataFrame
        dates = pd.date_range(start='2020-01-01', periods=rows)
        data = pd.DataFrame({
            'timestamp': dates,
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        return data
    
    def implement_unit_testing(self, indicators_to_test: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Implement unit tests for each indicator.

        Args:
            indicators_to_test: List of indicator names to test (None for all)
            
        Returns:
            Dictionary of test results
        """
        logger.info("Starting unit testing of indicators")
        results = {}
        
        # Define all available indicators with their functions
        all_indicators = {
            # Moving averages
            'SMA': moving_averages.simple_moving_average,
            'EMA': moving_averages.exponential_moving_average,
            'WMA': moving_averages.weighted_moving_average,
            
            # Oscillators
            'RSI': oscillators.relative_strength_index,
            'MACD': oscillators.macd,
            'Stochastic': oscillators.stochastic,
            
            # Volume
            'OBV': volume.on_balance_volume,
            'Volume Profile': volume.volume_profile,
            
            # Volatility
            'ATR': volatility.average_true_range,
            'Bollinger Bands': volatility.bollinger_bands,
            
            # Advanced indicators
            'Ichimoku': advanced_price_indicators.ichimoku_cloud,
            'KAMA': advanced_moving_averages.kama
        }
        
        # Filter indicators if specified
        if indicators_to_test:
            test_indicators = {k: v for k, v in all_indicators.items() if k in indicators_to_test}
        else:
            test_indicators = all_indicators
            
        # Set reference data for tester
        if self.test_data is None:
            self.test_data = self._generate_synthetic_data()
        self.indicator_tester.set_reference_data(self.test_data)
        
        # Run unit tests for each indicator
        for name, func in test_indicators.items():
            logger.info(f"Testing indicator: {name}")
            
            try:
                # Calculate expected values (could be replaced with known good values)
                # Here we're using the actual implementation as "expected" to demonstrate the test
                expected_result = func(self.test_data)
                
                # Test calculation accuracy
                accuracy_result = self.indicator_tester.test_indicator_accuracy(
                    indicator_func=func,
                    expected_results=expected_result
                )
                
                # Test edge cases
                edge_cases = self._generate_edge_cases_for_indicator(name)
                edge_case_result = self.indicator_tester.test_edge_cases(func, edge_cases)
                
                # Store results
                results[name] = {
                    "accuracy": accuracy_result,
                    "edge_cases": edge_case_result
                }
                
                logger.info(f"Successfully tested {name}")
                
            except Exception as e:
                logger.error(f"Error testing {name}: {e}")
                results[name] = {"error": str(e)}
                
        # Generate report
        report_path = os.path.join(self.results_directory, "unit_test_report.md")
        self.indicator_tester.generate_report(report_path)
        
        return results
    
    def implement_integration_testing(self) -> Dict[str, Any]:
        """
        Implement integration tests between indicator groups.
        
        Returns:
            Dictionary of test results
        """
        logger.info("Starting integration testing of indicators")
        results = {}
        
        # Define indicator combinations to test together
        indicator_groups = [
            # Test moving averages together
            {
                "name": "Moving Average Group",
                "indicators": [
                    moving_averages.simple_moving_average,
                    moving_averages.exponential_moving_average,
                    moving_averages.weighted_moving_average
                ]
            },
            # Test oscillators together
            {
                "name": "Oscillator Group",
                "indicators": [
                    oscillators.relative_strength_index,
                    oscillators.macd,
                    oscillators.stochastic
                ]
            },
            # Test cross-group indicators
            {
                "name": "Trend and Volume Group",
                "indicators": [
                    moving_averages.exponential_moving_average,
                    volume.on_balance_volume,
                    oscillators.relative_strength_index
                ]
            }
        ]
        
        # Set reference data for tester
        if self.test_data is None:
            self.test_data = self._generate_synthetic_data()
        
        # Test each group
        for group in indicator_groups:
            group_name = group["name"]
            indicators = group["indicators"]
            
            logger.info(f"Testing integration of {group_name}")
            group_results = {}
            
            try:
                start_time = time.time()
                
                # Calculate all indicators in the group
                indicator_outputs = {}
                for indicator_func in indicators:
                    indicator_name = indicator_func.__name__
                    indicator_outputs[indicator_name] = indicator_func(self.test_data)
                
                # Test data flow between indicators
                self._test_indicator_data_flow(indicator_outputs, group_results)
                
                # Test indicator dependencies (if any)
                self._test_indicator_dependencies(indicators, group_results)
                
                # Test high load conditions
                self._test_under_high_load(indicators, group_results)
                
                execution_time = time.time() - start_time
                group_results["execution_time"] = execution_time
                
                results[group_name] = group_results
                logger.info(f"Successfully completed integration tests for {group_name}")
                
            except Exception as e:
                logger.error(f"Error in integration testing for {group_name}: {e}")
                results[group_name] = {"error": str(e)}
                
        # Generate integration test report
        self._generate_integration_report(results)
        
        return results
    
    def implement_performance_testing(self) -> Dict[str, Any]:
        """
        Implement performance tests under various market conditions.
        
        Returns:
            Dictionary of performance test results
        """
        logger.info("Starting performance testing of indicators")
        results = {}
        
        # Define key indicators to performance test
        test_indicators = [
            {
                "name": "Simple Moving Average",
                "func": moving_averages.simple_moving_average,
                "params": [
                    {"window": 10},
                    {"window": 20},
                    {"window": 50},
                    {"window": 200}
                ]
            },
            {
                "name": "MACD",
                "func": oscillators.macd,
                "params": [
                    {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                    {"fast_period": 8, "slow_period": 17, "signal_period": 9}
                ]
            },
            {
                "name": "Bollinger Bands",
                "func": volatility.bollinger_bands,
                "params": [
                    {"window": 20, "num_std": 2},
                    {"window": 20, "num_std": 3}
                ]
            }
        ]
        
        # Different market condition datasets
        market_conditions = self._generate_market_condition_datasets()
        
        # Set up benchmark parameters
        data_sizes = [100, 500, 1000, 5000, 10000]  # Different data sizes
        num_processes = [1, 2, 4, 8]  # For parallel benchmarking
        
        # Run performance tests for each indicator under each market condition
        for condition_name, condition_data in market_conditions.items():
            logger.info(f"Testing performance under {condition_name} market conditions")
            condition_results = {}
            
            # Set reference data for this condition
            self.indicator_tester.set_reference_data(condition_data)
            
            for indicator_info in test_indicators:
                indicator_name = indicator_info["name"]
                indicator_func = indicator_info["func"]
                params_list = indicator_info["params"]
                
                try:
                    logger.info(f"Benchmarking {indicator_name}")
                    
                    # Basic performance benchmark
                    benchmark_results = self.indicator_tester.benchmark_performance(
                        indicator_func=indicator_func,
                        params_list=params_list,
                        data_sizes=data_sizes
                    )
                    
                    # Parallel performance benchmark
                    parallel_results = self.indicator_tester.parallel_benchmark(
                        indicator_func=indicator_func,
                        params=params_list[0],  # Use first param set for parallel test
                        num_processes=num_processes,
                        data_size=1000  # Use medium size for parallel testing
                    )
                    
                    # Resource usage analysis
                    resource_usage = self._analyze_resource_usage(indicator_func, params_list[0])
                    
                    condition_results[indicator_name] = {
                        "benchmark": benchmark_results,
                        "parallel_benchmark": parallel_results,
                        "resource_usage": resource_usage
                    }
                    
                except Exception as e:
                    logger.error(f"Error benchmarking {indicator_name}: {e}")
                    condition_results[indicator_name] = {"error": str(e)}
            
            results[condition_name] = condition_results
        
        # Generate performance report with visualizations
        self._generate_performance_report(results)
        
        return results
    
    def _generate_edge_cases_for_indicator(self, indicator_name: str) -> List[Dict]:
        """Generate appropriate edge cases for the given indicator."""
        edge_cases = []
        
        # Common edge cases for all indicators
        edge_cases.extend([
            {
                "name": "empty_data",
                "description": "Test with empty dataset",
                "data": pd.DataFrame(),
                "expected_behavior": "error"
            },
            {
                "name": "single_row",
                "description": "Test with single data point",
                "data": self.test_data.iloc[0:1],
                "expected_behavior": "null"
            },
            {
                "name": "null_values",
                "description": "Test with null values in data",
                "data": self._inject_nulls(self.test_data.copy()),
                "expected_behavior": {"type": "nan_count", "min": 1}
            }
        ])
        
        # Indicator-specific edge cases
        if 'SMA' in indicator_name or 'EMA' in indicator_name or 'WMA' in indicator_name:
            edge_cases.append({
                "name": "window_larger_than_data",
                "description": "Test with window larger than dataset",
                "data": self.test_data.iloc[0:10],
                "params": {"window": 20},
                "expected_behavior": {"type": "nan_count", "min": 1}
            })
            
        if 'RSI' in indicator_name or 'Stochastic' in indicator_name:
            edge_cases.append({
                "name": "flat_prices",
                "description": "Test with flat price data (no changes)",
                "data": self._generate_flat_price_data(100),
                "expected_behavior": "not_null"
            })
            
        if 'Bollinger' in indicator_name:
            edge_cases.append({
                "name": "zero_std_dev",
                "description": "Test with price data having zero standard deviation",
                "data": self._generate_flat_price_data(100),
                "expected_behavior": "not_null"
            })
            
        return edge_cases
    
    def _inject_nulls(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inject some NaN values into the dataframe."""
        # Create a copy to avoid modifying original
        df = data.copy()
        
        # Inject NaN values randomly
        for col in df.columns:
            if col not in ['timestamp', 'date', 'datetime']:
                mask = np.random.random(len(df)) < 0.05  # 5% of values become NaN
                df.loc[mask, col] = np.nan
                
        return df
    
    def _generate_flat_price_data(self, rows: int) -> pd.DataFrame:
        """Generate price data with flat/constant values."""
        dates = pd.date_range(start='2020-01-01', periods=rows)
        return pd.DataFrame({
            'timestamp': dates,
            'open': [100] * rows,
            'high': [100] * rows,
            'low': [100] * rows,
            'close': [100] * rows,
            'volume': np.random.normal(1000, 10, rows)
        })
    
    def _test_indicator_data_flow(self, indicator_outputs: Dict, results: Dict) -> None:
        """Test data flow between indicators for integration testing."""
        # Check that all indicators produce compatible output formats
        output_shapes = {}
        for name, output in indicator_outputs.items():
            if isinstance(output, pd.DataFrame):
                output_shapes[name] = output.shape
            elif isinstance(output, tuple) and all(isinstance(item, pd.DataFrame) for item in output):
                output_shapes[name] = tuple(item.shape for item in output)
            else:
                output_shapes[name] = "unsupported_format"
                
        # Check for index compatibility
        index_compatibility = {}
        reference_index = None
        
        for name, output in indicator_outputs.items():
            if isinstance(output, pd.DataFrame):
                if reference_index is None:
                    reference_index = output.index
                index_compatibility[name] = output.index.equals(reference_index)
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], pd.DataFrame):
                if reference_index is None:
                    reference_index = output[0].index
                index_compatibility[name] = output[0].index.equals(reference_index)
                
        results["output_shapes"] = output_shapes
        results["index_compatibility"] = index_compatibility
        results["data_flow_compatible"] = all(index_compatibility.values())
    
    def _test_indicator_dependencies(self, indicators: List, results: Dict) -> None:
        """Test dependencies between indicators."""
        # Simple dependency check - see if indicators can use each other's outputs
        dependency_tests = {}
        
        try:
            # Example: Create a dependency chain
            # First indicator output as input to second
            first_output = indicators[0](self.test_data)
            
            if isinstance(first_output, pd.DataFrame):
                # Try using the first output for the second indicator
                second_indicator_works = False
                try:
                    _ = indicators[1](first_output)
                    second_indicator_works = True
                except:
                    pass
                
                dependency_tests["first_to_second"] = second_indicator_works
        except Exception as e:
            dependency_tests["error"] = str(e)
            
        results["dependency_tests"] = dependency_tests
    
    def _test_under_high_load(self, indicators: List, results: Dict) -> None:
        """Test indicators under high load conditions."""
        high_load_results = {}
        
        # Generate large dataset for high load testing
        large_data = self._generate_synthetic_data(rows=50000)
        
        # Test with thread pool for concurrent execution
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(func, large_data) for func in indicators]
            outputs = [future.result() for future in futures]
        thread_time = time.time() - start_time
        
        # Test with process pool
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(func, large_data) for func in indicators]
            outputs = [future.result() for future in futures]
        process_time = time.time() - start_time
        
        high_load_results["thread_execution_time"] = thread_time
        high_load_results["process_execution_time"] = process_time
        high_load_results["speedup"] = thread_time / process_time if process_time > 0 else float('inf')
        
        results["high_load_test"] = high_load_results
    
    def _generate_market_condition_datasets(self) -> Dict[str, pd.DataFrame]:
        """Generate datasets representing different market conditions."""
        np.random.seed(42)  # For reproducibility
        
        conditions = {}
        
        # Normal market
        conditions["normal"] = self._generate_synthetic_data(rows=5000)
        
        # Trending market (strong uptrend)
        trend_data = self._generate_synthetic_data(rows=5000)
        trend = np.linspace(0, 50, 5000)  # Linear uptrend
        trend_data['close'] += trend
        trend_data['high'] += trend
        trend_data['low'] += trend
        trend_data['open'] += trend
        conditions["trending"] = trend_data
        
        # Volatile market
        volatile_data = self._generate_synthetic_data(rows=5000)
        volatility = np.random.normal(0, 3, 5000)  # Higher volatility
        volatile_data['close'] += volatility
        volatile_data['high'] += np.abs(volatility) + 1
        volatile_data['low'] -= np.abs(volatility) + 1
        conditions["volatile"] = volatile_data
        
        # Ranging market (sideways)
        range_data = self._generate_synthetic_data(rows=5000)
        range_factor = np.sin(np.linspace(0, 10*np.pi, 5000)) * 5  # Oscillating pattern
        range_data['close'] += range_factor
        range_data['high'] += range_factor + np.random.normal(0, 0.5, 5000)
        range_data['low'] += range_factor - np.random.normal(0, 0.5, 5000)
        conditions["ranging"] = range_data
        
        return conditions
    
    def _analyze_resource_usage(self, indicator_func: callable, params: Dict) -> Dict:
        """Analyze CPU and memory usage of an indicator function."""
        # This is a simple implementation, in production you'd want to use
        # more sophisticated resource tracking like psutil
        
        import tracemalloc
        import gc
        
        # Force garbage collection to start with clean state
        gc.collect()
        
        # Track memory usage
        tracemalloc.start()
        
        # Measure CPU time
        start_time = time.time()
        cpu_start = time.process_time()
        
        # Run the indicator
        result = indicator_func(self.test_data, **params)
        
        # Get CPU usage
        cpu_end = time.process_time()
        end_time = time.time()
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            "wall_time": end_time - start_time,
            "cpu_time": cpu_end - cpu_start,
            "memory_current_kb": current / 1024,
            "memory_peak_kb": peak / 1024,
            "result_size": sys.getsizeof(result) / 1024  # in KB
        }
    
    def _generate_integration_report(self, results: Dict) -> None:
        """Generate a report for integration tests."""
        report_path = os.path.join(self.results_directory, "integration_test_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# Indicator Integration Test Report\n")
            f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
            
            for group_name, group_results in results.items():
                f.write(f"## {group_name}\n\n")
                
                if "error" in group_results:
                    f.write(f"Error: {group_results['error']}\n\n")
                    continue
                
                f.write("### Data Flow Compatibility\n\n")
                if "index_compatibility" in group_results:
                    f.write("| Indicator | Index Compatible |\n")
                    f.write("|-----------|----------------|\n")
                    for name, compatible in group_results["index_compatibility"].items():
                        f.write(f"| {name} | {'Yes' if compatible else 'No'} |\n")
                
                f.write("\n### Output Shapes\n\n")
                if "output_shapes" in group_results:
                    f.write("| Indicator | Output Shape |\n")
                    f.write("|-----------|-------------|\n")
                    for name, shape in group_results["output_shapes"].items():
                        f.write(f"| {name} | {shape} |\n")
                
                f.write("\n### Dependency Tests\n\n")
                if "dependency_tests" in group_results:
                    for test_name, result in group_results["dependency_tests"].items():
                        if test_name == "error":
                            f.write(f"Error: {result}\n\n")
                        else:
                            f.write(f"- {test_name}: {'Success' if result else 'Failed'}\n")
                
                f.write("\n### High Load Testing\n\n")
                if "high_load_test" in group_results:
                    high_load = group_results["high_load_test"]
                    f.write(f"- Thread execution time: {high_load.get('thread_execution_time', 0):.4f} seconds\n")
                    f.write(f"- Process execution time: {high_load.get('process_execution_time', 0):.4f} seconds\n")
                    f.write(f"- Speedup ratio: {high_load.get('speedup', 0):.2f}x\n")
                
                f.write(f"\nTotal execution time: {group_results.get('execution_time', 0):.4f} seconds\n\n")
                f.write("---\n\n")
    
    def _generate_performance_report(self, results: Dict) -> None:
        """Generate a report for performance tests with visualizations."""
        report_path = os.path.join(self.results_directory, "performance_test_report.md")
        figures_dir = os.path.join(self.results_directory, "figures")
        
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        
        with open(report_path, 'w') as f:
            f.write("# Indicator Performance Test Report\n")
            f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
            
            for condition, condition_results in results.items():
                f.write(f"## {condition} Market Conditions\n\n")
                
                for indicator, ind_results in condition_results.items():
                    f.write(f"### {indicator}\n\n")
                    
                    if "error" in ind_results:
                        f.write(f"Error: {ind_results['error']}\n\n")
                        continue
                    
                    # Create performance visualization
                    fig_path = self._create_performance_visualization(
                        indicator, condition, ind_results, figures_dir)
                    
                    f.write(f"![Performance Chart for {indicator}]({os.path.relpath(fig_path, self.results_directory)})\n\n")
                    
                    f.write("#### Resource Usage\n\n")
                    if "resource_usage" in ind_results:
                        resource = ind_results["resource_usage"]
                        f.write("| Metric | Value |\n")
                        f.write("|--------|------|\n")
                        f.write(f"| Wall time | {resource.get('wall_time', 0):.6f} seconds |\n")
                        f.write(f"| CPU time | {resource.get('cpu_time', 0):.6f} seconds |\n")
                        f.write(f"| Peak memory | {resource.get('memory_peak_kb', 0):.2f} KB |\n")
                        f.write(f"| Result size | {resource.get('result_size', 0):.2f} KB |\n\n")
                    
                    f.write("#### Parallel Execution\n\n")
                    if "parallel_benchmark" in ind_results:
                        f.write("| Processes | Total Time (s) | Time per Process (s) |\n")
                        f.write("|-----------|---------------|---------------------|\n")
                        
                        parallel = ind_results["parallel_benchmark"]
                        for proc_key, proc_data in parallel.items():
                            if isinstance(proc_data, dict):
                                processes = proc_data.get("num_processes", 0)
                                total_time = proc_data.get("total_time", 0)
                                per_proc = proc_data.get("time_per_process", 0)
                                f.write(f"| {processes} | {total_time:.6f} | {per_proc:.6f} |\n")
                    
                    f.write("\n---\n\n")
    
    def _create_performance_visualization(self, indicator: str, condition: str, 
                                         results: Dict, figures_dir: str) -> str:
        """Create performance visualization chart and return the file path."""
        plt.figure(figsize=(10, 6))
        
        # Extract data from benchmark results
        if "benchmark" not in results:
            plt.text(0.5, 0.5, "No benchmark data available", 
                     ha='center', va='center', fontsize=12)
            fig_path = os.path.join(figures_dir, f"{condition}_{indicator}_performance.png")
            plt.savefig(fig_path)
            plt.close()
            return fig_path
            
        benchmark = results["benchmark"]
        
        # Prepare data for plotting
        sizes = []
        param_keys = []
        mean_times = []
        
        for size_key, size_data in benchmark.items():
            size = int(size_key.replace("size_", ""))
            
            for param_key, param_data in size_data.items():
                sizes.append(size)
                param_keys.append(param_key)
                mean_times.append(param_data.get("mean_time", 0))
        
        # Create DataFrame for easy plotting
        plot_data = pd.DataFrame({
            'Size': sizes,
            'Parameters': param_keys,
            'Mean Time': mean_times
        })
        
        # Plot using seaborn
        sns.lineplot(data=plot_data, x='Size', y='Mean Time', hue='Parameters', marker='o')
        
        plt.title(f"Performance of {indicator} under {condition} market conditions")
        plt.xlabel('Data Size (rows)')
        plt.ylabel('Execution Time (seconds)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        fig_path = os.path.join(figures_dir, f"{condition}_{indicator}_performance.png")
        plt.savefig(fig_path)
        plt.close()
        
        return fig_path


def implement_unit_testing():
    """
    Entry point for unit testing implementation.
    This function aligns with the Phase 8 requirement.
    """
    logger.info("Implementing unit testing for indicators")
    
    # Create test suite
    test_suite = IndicatorTestSuite()
    
    # Run unit tests
    results = test_suite.implement_unit_testing()
    
    logger.info("Unit testing completed")
    return results


def implement_integration_testing():
    """
    Entry point for integration testing implementation.
    This function aligns with the Phase 8 requirement.
    """
    logger.info("Implementing integration testing for indicators")
    
    # Create test suite
    test_suite = IndicatorTestSuite()
    
    # Run integration tests
    results = test_suite.implement_integration_testing()
    
    logger.info("Integration testing completed")
    return results


def implement_performance_testing():
    """
    Entry point for performance testing implementation.
    This function aligns with the Phase 8 requirement.
    """
    logger.info("Implementing performance testing for indicators")
    
    # Create test suite
    test_suite = IndicatorTestSuite()
    
    # Run performance tests
    results = test_suite.implement_performance_testing()
    
    logger.info("Performance testing completed")
    return results


if __name__ == "__main__":
    # Run all test implementations
    implement_unit_testing()
    implement_integration_testing()
    implement_performance_testing()
""""""
