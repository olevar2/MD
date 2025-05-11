"""
Performance Enhancement Testing Framework.

This module provides tools for testing and benchmarking the performance optimization
modules when applied to technical indicators.
"""
import pandas as pd
import numpy as np
import time
import unittest
import logging
import os
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
import matplotlib.pyplot as plt
from contextlib import contextmanager
import gc
import tempfile
import json
import psutil
from feature_store_service.indicators.base_indicator import BaseIndicator
from feature_store_service.indicators.performance_enhanced_indicator import PerformanceEnhancedIndicator
from feature_store_service.optimization.gpu_acceleration import is_gpu_available, GPUAccelerator
from feature_store_service.optimization.advanced_calculation import smart_cache
from feature_store_service.optimization.load_balancing import get_load_balancer, initialize_load_balancer
from feature_store_service.optimization.memory_optimization import MemoryOptimizer
logger = logging.getLogger(__name__)

@contextmanager
def timing_context(description: str):
    """
    Context manager for timing code execution.
    
    Args:
        description: Description of the operation being timed
    """
    start_time = time.time()
    exception = None
    try:
        yield
    except Exception as e:
        exception = e
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f'Timing: {description} completed in {elapsed_time:.4f}s')
    if exception:
        raise exception

class PerformanceTestResult:
    """Data class for storing performance test results."""

    def __init__(self, name: str, standard_time: float, enhanced_time: float, standard_memory: float=None, enhanced_memory: float=None):
        """
        Initialize performance test result.
        
        Args:
            name: Test name
            standard_time: Execution time for standard implementation (seconds)
            enhanced_time: Execution time for enhanced implementation (seconds)
            standard_memory: Memory usage for standard implementation (MB)
            enhanced_memory: Memory usage for enhanced implementation (MB)
        """
        self.name = name
        self.standard_time = standard_time
        self.enhanced_time = enhanced_time
        self.standard_memory = standard_memory
        self.enhanced_memory = enhanced_memory
        self.speedup = standard_time / enhanced_time if enhanced_time > 0 else float('inf')
        self.memory_reduction = (standard_memory / enhanced_memory if enhanced_memory > 0 else float('inf')) if standard_memory and enhanced_memory else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {'name': self.name, 'standard_time': self.standard_time, 'enhanced_time': self.enhanced_time, 'standard_memory': self.standard_memory, 'enhanced_memory': self.enhanced_memory, 'speedup': self.speedup, 'memory_reduction': self.memory_reduction}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceTestResult':
        """Create from dictionary."""
        return cls(name=data['name'], standard_time=data['standard_time'], enhanced_time=data['enhanced_time'], standard_memory=data.get('standard_memory'), enhanced_memory=data.get('enhanced_memory'))

class IndicatorPerformanceBenchmark:
    """
    Benchmarking tool for comparing standard and enhanced indicator performance.
    """

    def __init__(self, output_dir: Optional[str]=None):
        """
        Initialize the benchmark tool.
        
        Args:
            output_dir: Directory for saving results (None for temp directory)
        """
        self.output_dir = output_dir or tempfile.mkdtemp()
        self.results = []

    def measure_memory(self) -> float:
        """
        Measure current memory usage.
        
        Returns:
            Current memory usage in MB
        """
        gc.collect()
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)

    def benchmark_indicator(self, standard_indicator: BaseIndicator, enhanced_indicator: PerformanceEnhancedIndicator, test_data: pd.DataFrame, name: str=None) -> PerformanceTestResult:
        """
        Benchmark and compare standard vs enhanced indicator performance.
        
        Args:
            standard_indicator: Standard indicator instance
            enhanced_indicator: Enhanced indicator instance
            test_data: Test data for benchmarking
            name: Test name (defaults to indicator class names)
            
        Returns:
            Performance test result
        """
        if name is None:
            name = f'{standard_indicator.__class__.__name__} vs {enhanced_indicator.__class__.__name__}'
        logger.info(f'Starting benchmark: {name}')
        if hasattr(smart_cache, 'clear'):
            smart_cache.clear()
        standard_start_memory = self.measure_memory()
        standard_start_time = time.time()
        standard_result = standard_indicator.calculate(test_data)
        standard_time = time.time() - standard_start_time
        standard_memory = self.measure_memory() - standard_start_memory
        standard_result = None
        gc.collect()
        enhanced_start_memory = self.measure_memory()
        enhanced_start_time = time.time()
        enhanced_result = enhanced_indicator.calculate(test_data)
        enhanced_time = time.time() - enhanced_start_time
        enhanced_memory = self.measure_memory() - enhanced_start_memory
        result = PerformanceTestResult(name=name, standard_time=standard_time, enhanced_time=enhanced_time, standard_memory=standard_memory, enhanced_memory=enhanced_memory)
        self.results.append(result)
        logger.info(f'Benchmark result: {name}')
        logger.info(f'  Standard execution time: {standard_time:.4f}s')
        logger.info(f'  Enhanced execution time: {enhanced_time:.4f}s')
        logger.info(f'  Speedup: {result.speedup:.2f}x')
        if standard_memory is not None and enhanced_memory is not None:
            logger.info(f'  Standard memory usage: {standard_memory:.2f}MB')
            logger.info(f'  Enhanced memory usage: {enhanced_memory:.2f}MB')
            logger.info(f'  Memory reduction: {result.memory_reduction:.2f}x')
        return result

    def benchmark_multiple_data_sizes(self, standard_indicator_factory: Callable[[], BaseIndicator], enhanced_indicator_factory: Callable[[], PerformanceEnhancedIndicator], data_generator: Callable[[int], pd.DataFrame], sizes: List[int], name_prefix: str='') -> List[PerformanceTestResult]:
        """
        Run benchmarks with multiple data sizes.
        
        Args:
            standard_indicator_factory: Function to create standard indicator instances
            enhanced_indicator_factory: Function to create enhanced indicator instances
            data_generator: Function to generate test data of specified size
            sizes: List of data sizes to test
            name_prefix: Prefix for test names
            
        Returns:
            List of performance test results
        """
        results = []
        for size in sizes:
            standard_indicator = standard_indicator_factory()
            enhanced_indicator = enhanced_indicator_factory()
            test_data = data_generator(size)
            name = f'{name_prefix}Size_{size}'
            result = self.benchmark_indicator(standard_indicator, enhanced_indicator, test_data, name)
            results.append(result)
            standard_indicator = None
            enhanced_indicator = None
            test_data = None
            gc.collect()
        self.results.extend(results)
        return results

    def save_results(self, filename: str='benchmark_results.json'):
        """
        Save benchmark results to file.
        
        Args:
            filename: Output filename
        """
        os.makedirs(self.output_dir, exist_ok=True)
        result_dicts = [result.to_dict() for result in self.results]
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(result_dicts, f, indent=2)
        logger.info(f'Saved benchmark results to {filepath}')

    def load_results(self, filepath: str) -> List[PerformanceTestResult]:
        """
        Load benchmark results from file.
        
        Args:
            filepath: Input filepath
            
        Returns:
            List of performance test results
        """
        with open(filepath, 'r') as f:
            result_dicts = json.load(f)
        results = [PerformanceTestResult.from_dict(d) for d in result_dicts]
        self.results = results
        return results

    def plot_results(self, metric: str='speedup', save_path: Optional[str]=None):
        """
        Plot benchmark results.
        
        Args:
            metric: Metric to plot ('speedup' or 'memory_reduction')
            save_path: Path to save the plot (None for display only)
        """
        if not self.results:
            logger.warning('No results to plot')
            return
        result_groups = {}
        for result in self.results:
            parts = result.name.split('Size_')
            if len(parts) == 2:
                prefix, size = parts
                size = int(size)
                if prefix not in result_groups:
                    result_groups[prefix] = {'sizes': [], 'values': []}
                result_groups[prefix]['sizes'].append(size)
                if metric == 'speedup':
                    value = result.speedup
                elif metric == 'memory_reduction':
                    value = result.memory_reduction or 1.0
                else:
                    raise ValueError(f'Unknown metric: {metric}')
                result_groups[prefix]['values'].append(value)
        plt.figure(figsize=(10, 6))
        if result_groups:
            for prefix, data in result_groups.items():
                sizes = data['sizes']
                values = data['values']
                sizes, values = zip(*sorted(zip(sizes, values)))
                plt.plot(sizes, values, 'o-', label=prefix.strip())
            plt.xscale('log')
            plt.xlabel('Data Size (points)')
        else:
            names = [r.name for r in self.results]
            if metric == 'speedup':
                values = [r.speedup for r in self.results]
                plt.ylabel('Speedup Factor (higher is better)')
            else:
                values = [r.memory_reduction or 1.0 for r in self.results]
                plt.ylabel('Memory Reduction Factor (higher is better)')
            plt.bar(names, values)
            plt.xticks(rotation=45, ha='right')
        plt.title(f'Performance Enhancement: {metric.replace('_', ' ').title()}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info(f'Saved plot to {save_path}')
        else:
            plt.show()

class PerformanceTests(unittest.TestCase):
    """
    Test suite for performance-enhanced indicators.
    """

    def set_up(self):
        """Set up test environment."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=10000, freq='1min')
        self.test_data = pd.DataFrame({'datetime': dates, 'open': np.random.randn(10000).cumsum() + 100, 'high': np.random.randn(10000).cumsum() + 102, 'low': np.random.randn(10000).cumsum() + 98, 'close': np.random.randn(10000).cumsum() + 100, 'volume': np.random.randint(100, 10000, size=10000)}).set_index('datetime')
        initialize_load_balancer()

    def tear_down(self):
        """Clean up after tests."""
        if is_gpu_available():
            try:
                if hasattr(self, 'gpu_accelerator'):
                    del self.gpu_accelerator
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
        gc.collect()

    def test_gpu_acceleration(self):
        """Test GPU acceleration module."""
        if not is_gpu_available():
            self.skipTest('GPU not available for testing')
        accelerator = GPUAccelerator()
        test_array = np.random.randn(1000, 1000)
        gpu_array = accelerator.to_gpu(test_array)
        cpu_array = accelerator.to_cpu(gpu_array)
        np.testing.assert_allclose(test_array, cpu_array, rtol=1e-05)
        with timing_context('GPU Moving Average'):
            ma_gpu = accelerator.compute_moving_average(test_array[:, 0], 20)
        with timing_context('CPU Moving Average'):
            ma_cpu = np.full_like(test_array[:, 0], np.nan)
            for i in range(20 - 1, len(test_array)):
                ma_cpu[i] = np.mean(test_array[i - 20 + 1:i + 1, 0])
        np.testing.assert_allclose(ma_gpu[20 - 1:], ma_cpu[20 - 1:], rtol=1e-05)

    def test_memory_optimization(self):
        """Test memory optimization module."""
        large_df = pd.DataFrame({'float_col': np.random.randn(100000), 'int_col': np.random.randint(0, 100, size=100000), 'category_col': np.random.choice(['A', 'B', 'C', 'D'], size=100000), 'sparse_col': np.random.choice([0, 1], p=[0.9, 0.1], size=100000)})
        original_size = large_df.memory_usage(deep=True).sum() / (1024 * 1024)
        with timing_context('Memory Optimization'):
            optimized_df = MemoryOptimizer.optimize_dataframe(large_df)
        optimized_size = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
        self.assertLess(optimized_size, original_size * 0.8, f'Memory not sufficiently reduced: {original_size:.2f}MB -> {optimized_size:.2f}MB')
        logger.info(f'Memory optimization: {original_size:.2f}MB -> {optimized_size:.2f}MB ({(1 - optimized_size / original_size) * 100:.1f}% reduction)')
        pd.testing.assert_frame_equal(large_df, optimized_df)

    def generate_benchmark_report(self, output_dir: str='benchmark_results'):
        """
        Generate comprehensive performance benchmark report.
        
        Args:
            output_dir: Directory for saving results
        """
        from feature_store_service.indicators.moving_averages import SMA
        from feature_store_service.indicators.advanced.enhanced_sma import EnhancedSMA
        benchmark = IndicatorPerformanceBenchmark(output_dir=output_dir)

        def generate_data(size):
            dates = pd.date_range(start='2020-01-01', periods=size, freq='1min')
            return pd.DataFrame({'datetime': dates, 'close': np.random.randn(size).cumsum() + 100}).set_index('datetime')
        sizes = [1000, 10000, 100000]
        benchmark.benchmark_multiple_data_sizes(lambda: SMA(window=20), lambda: EnhancedSMA(window=20, use_gpu=is_gpu_available()), generate_data, sizes, name_prefix='SMA_')
        benchmark.save_results()
        benchmark.plot_results(save_path=os.path.join(output_dir, 'speedup_chart.png'))
        benchmark.plot_results(metric='memory_reduction', save_path=os.path.join(output_dir, 'memory_chart.png'))
        html_report = f'\n        <!DOCTYPE html>\n        <html>\n        <head>\n            <title>Performance Optimization Benchmark Report</title>\n            <style>\n                body {{ font-family: Arial, sans-serif; margin: 20px; }}\n                h1, h2 {{ color: #333; }}\n                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}\n                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}\n                th {{ background-color: #f2f2f2; text-align: center; }}\n                .better {{ color: green; font-weight: bold; }}\n                .center {{ text-align: center; }}\n                img {{ max-width: 100%; height: auto; margin: 20px 0; }}\n                .summary {{ background-color: #f8f8f8; padding: 10px; border-radius: 5px; }}\n            </style>\n        </head>\n        <body>\n            <h1>Performance Optimization Benchmark Report</h1>\n            \n            <div class="summary">\n                <h2>Summary</h2>\n                <p>This report shows the performance comparison between standard and enhanced indicator implementations.</p>\n                <p>Enhanced implementations use GPU acceleration, advanced calculation techniques, load balancing, and memory optimization.</p>\n                <p>GPU Available: {is_gpu_available()}</p>\n            </div>\n            \n            <h2>Performance Results</h2>\n            <table>\n                <tr>\n                    <th>Test</th>\n                    <th>Standard Time (s)</th>\n                    <th>Enhanced Time (s)</th>\n                    <th>Speedup Factor</th>\n                    <th>Memory Reduction</th>\n                </tr>\n        '
        for result in benchmark.results:
            html_report += f'\n                <tr>\n                    <td>{result.name}</td>\n                    <td>{result.standard_time:.4f}</td>\n                    <td>{result.enhanced_time:.4f}</td>\n                    <td class="better">{result.speedup:.2f}x</td>\n                    <td class="{('better' if result.memory_reduction and result.memory_reduction > 1.0 else '')}">\n                        {(f'{result.memory_reduction:.2f}x' if result.memory_reduction else 'N/A')}\n                    </td>\n                </tr>\n            '
        html_report += '\n            </table>\n            \n            <h2>Performance Charts</h2>\n            \n            <h3>Execution Speed Improvement</h3>\n            <img src="speedup_chart.png" alt="Speedup Chart">\n            \n            <h3>Memory Usage Improvement</h3>\n            <img src="memory_chart.png" alt="Memory Reduction Chart">\n        </body>\n        </html>\n        '
        report_path = os.path.join(output_dir, 'benchmark_report.html')
        with open(report_path, 'w') as f:
            f.write(html_report)
        logger.info(f'Generated benchmark report at {report_path}')
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    unittest.main()