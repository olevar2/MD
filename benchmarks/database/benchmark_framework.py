"""
Benchmark framework for database utilities.

This module provides a framework for benchmarking database utilities in the forex trading platform.
"""
import os
import sys
import time
import logging
import asyncio
import argparse
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from datetime import datetime, timedelta
import psutil
import gc
import tracemalloc
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BenchmarkResult:
    """Class to store benchmark results."""
    
    def __init__(self, name: str, category: str):
        """Initialize benchmark result."""
        self.name = name
        self.category = category
        self.execution_times = []
        self.memory_usage = []
        self.throughput = []
        self.start_time = None
        self.end_time = None
        self.metadata = {}
    
    def start(self):
        """Start the benchmark."""
        self.start_time = time.time()
    
    def end(self):
        """End the benchmark."""
        self.end_time = time.time()
    
    def add_execution_time(self, execution_time: float):
        """Add execution time to the result."""
        self.execution_times.append(execution_time)
    
    def add_memory_usage(self, memory_usage: float):
        """Add memory usage to the result."""
        self.memory_usage.append(memory_usage)
    
    def add_throughput(self, throughput: float):
        """Add throughput to the result."""
        self.throughput.append(throughput)
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the result."""
        self.metadata[key] = value
    
    @property
    def total_time(self) -> float:
        """Get the total time of the benchmark."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    @property
    def avg_execution_time(self) -> float:
        """Get the average execution time."""
        if not self.execution_times:
            return 0.0
        return sum(self.execution_times) / len(self.execution_times)
    
    @property
    def min_execution_time(self) -> float:
        """Get the minimum execution time."""
        if not self.execution_times:
            return 0.0
        return min(self.execution_times)
    
    @property
    def max_execution_time(self) -> float:
        """Get the maximum execution time."""
        if not self.execution_times:
            return 0.0
        return max(self.execution_times)
    
    @property
    def avg_memory_usage(self) -> float:
        """Get the average memory usage."""
        if not self.memory_usage:
            return 0.0
        return sum(self.memory_usage) / len(self.memory_usage)
    
    @property
    def avg_throughput(self) -> float:
        """Get the average throughput."""
        if not self.throughput:
            return 0.0
        return sum(self.throughput) / len(self.throughput)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "name": self.name,
            "category": self.category,
            "total_time": self.total_time,
            "avg_execution_time": self.avg_execution_time,
            "min_execution_time": self.min_execution_time,
            "max_execution_time": self.max_execution_time,
            "avg_memory_usage": self.avg_memory_usage,
            "avg_throughput": self.avg_throughput,
            "metadata": self.metadata,
        }


class DatabaseBenchmark:
    """Class to benchmark database utilities."""
    
    def __init__(self, name: str, output_dir: str = None):
        """Initialize database benchmark."""
        self.name = name
        self.output_dir = output_dir or os.path.join("benchmarks", "database", "results")
        self.results = {}
        self.baseline_results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def benchmark_sync(
        self,
        name: str,
        category: str,
        func: Callable,
        args: Tuple = None,
        kwargs: Dict[str, Any] = None,
        setup: Callable = None,
        teardown: Callable = None,
        repeat: int = 5,
        metadata: Dict[str, Any] = None,
    ) -> BenchmarkResult:
        """
        Benchmark a synchronous function.
        
        Args:
            name: Name of the benchmark
            category: Category of the benchmark
            func: Function to benchmark
            args: Arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
            setup: Setup function to call before each iteration
            teardown: Teardown function to call after each iteration
            repeat: Number of times to repeat the benchmark
            metadata: Additional metadata to store with the result
            
        Returns:
            Benchmark result
        """
        args = args or ()
        kwargs = kwargs or {}
        metadata = metadata or {}
        
        # Create result object
        result = BenchmarkResult(name, category)
        
        # Add metadata
        for key, value in metadata.items():
            result.add_metadata(key, value)
        
        # Start the benchmark
        result.start()
        
        # Run the benchmark
        for i in range(repeat):
            # Run setup if provided
            if setup:
                setup()
            
            # Clear memory
            gc.collect()
            
            # Start memory tracking
            tracemalloc.start()
            
            # Measure execution time
            start_time = time.time()
            func_result = func(*args, **kwargs)
            end_time = time.time()
            
            # Measure memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculate execution time
            execution_time = end_time - start_time
            
            # Add results
            result.add_execution_time(execution_time)
            result.add_memory_usage(peak / 1024 / 1024)  # Convert to MB
            
            # Run teardown if provided
            if teardown:
                teardown()
        
        # End the benchmark
        result.end()
        
        # Store the result
        self.results[name] = result
        
        return result
    
    async def benchmark_async(
        self,
        name: str,
        category: str,
        func: Callable,
        args: Tuple = None,
        kwargs: Dict[str, Any] = None,
        setup: Callable = None,
        teardown: Callable = None,
        repeat: int = 5,
        metadata: Dict[str, Any] = None,
    ) -> BenchmarkResult:
        """
        Benchmark an asynchronous function.
        
        Args:
            name: Name of the benchmark
            category: Category of the benchmark
            func: Async function to benchmark
            args: Arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
            setup: Setup function to call before each iteration
            teardown: Teardown function to call after each iteration
            repeat: Number of times to repeat the benchmark
            metadata: Additional metadata to store with the result
            
        Returns:
            Benchmark result
        """
        args = args or ()
        kwargs = kwargs or {}
        metadata = metadata or {}
        
        # Create result object
        result = BenchmarkResult(name, category)
        
        # Add metadata
        for key, value in metadata.items():
            result.add_metadata(key, value)
        
        # Start the benchmark
        result.start()
        
        # Run the benchmark
        for i in range(repeat):
            # Run setup if provided
            if setup:
                if asyncio.iscoroutinefunction(setup):
                    await setup()
                else:
                    setup()
            
            # Clear memory
            gc.collect()
            
            # Start memory tracking
            tracemalloc.start()
            
            # Measure execution time
            start_time = time.time()
            func_result = await func(*args, **kwargs)
            end_time = time.time()
            
            # Measure memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculate execution time
            execution_time = end_time - start_time
            
            # Add results
            result.add_execution_time(execution_time)
            result.add_memory_usage(peak / 1024 / 1024)  # Convert to MB
            
            # Run teardown if provided
            if teardown:
                if asyncio.iscoroutinefunction(teardown):
                    await teardown()
                else:
                    teardown()
        
        # End the benchmark
        result.end()
        
        # Store the result
        self.results[name] = result
        
        return result
    
    def benchmark_throughput(
        self,
        name: str,
        category: str,
        func: Callable,
        args_list: List[Tuple],
        kwargs_list: List[Dict[str, Any]] = None,
        setup: Callable = None,
        teardown: Callable = None,
        concurrency: int = 1,
        metadata: Dict[str, Any] = None,
    ) -> BenchmarkResult:
        """
        Benchmark throughput of a synchronous function.
        
        Args:
            name: Name of the benchmark
            category: Category of the benchmark
            func: Function to benchmark
            args_list: List of arguments to pass to the function
            kwargs_list: List of keyword arguments to pass to the function
            setup: Setup function to call before the benchmark
            teardown: Teardown function to call after the benchmark
            concurrency: Number of concurrent executions
            metadata: Additional metadata to store with the result
            
        Returns:
            Benchmark result
        """
        kwargs_list = kwargs_list or [{}] * len(args_list)
        metadata = metadata or {}
        
        # Create result object
        result = BenchmarkResult(name, category)
        
        # Add metadata
        for key, value in metadata.items():
            result.add_metadata(key, value)
        
        # Add metadata about the benchmark
        result.add_metadata("total_operations", len(args_list))
        result.add_metadata("concurrency", concurrency)
        
        # Run setup if provided
        if setup:
            setup()
        
        # Start the benchmark
        result.start()
        
        # Clear memory
        gc.collect()
        
        # Start memory tracking
        tracemalloc.start()
        
        # Measure execution time
        start_time = time.time()
        
        # Execute the function with each set of arguments
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for args, kwargs in zip(args_list, kwargs_list):
                futures.append(executor.submit(func, *args, **kwargs))
            
            # Wait for all futures to complete
            for future in futures:
                future.result()
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Measure memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate throughput (operations per second)
        throughput = len(args_list) / execution_time
        
        # Add results
        result.add_execution_time(execution_time)
        result.add_memory_usage(peak / 1024 / 1024)  # Convert to MB
        result.add_throughput(throughput)
        
        # Run teardown if provided
        if teardown:
            teardown()
        
        # End the benchmark
        result.end()
        
        # Store the result
        self.results[name] = result
        
        return result
    
    async def benchmark_async_throughput(
        self,
        name: str,
        category: str,
        func: Callable,
        args_list: List[Tuple],
        kwargs_list: List[Dict[str, Any]] = None,
        setup: Callable = None,
        teardown: Callable = None,
        concurrency: int = 1,
        metadata: Dict[str, Any] = None,
    ) -> BenchmarkResult:
        """
        Benchmark throughput of an asynchronous function.
        
        Args:
            name: Name of the benchmark
            category: Category of the benchmark
            func: Async function to benchmark
            args_list: List of arguments to pass to the function
            kwargs_list: List of keyword arguments to pass to the function
            setup: Setup function to call before the benchmark
            teardown: Teardown function to call after the benchmark
            concurrency: Number of concurrent executions
            metadata: Additional metadata to store with the result
            
        Returns:
            Benchmark result
        """
        kwargs_list = kwargs_list or [{}] * len(args_list)
        metadata = metadata or {}
        
        # Create result object
        result = BenchmarkResult(name, category)
        
        # Add metadata
        for key, value in metadata.items():
            result.add_metadata(key, value)
        
        # Add metadata about the benchmark
        result.add_metadata("total_operations", len(args_list))
        result.add_metadata("concurrency", concurrency)
        
        # Run setup if provided
        if setup:
            if asyncio.iscoroutinefunction(setup):
                await setup()
            else:
                setup()
        
        # Start the benchmark
        result.start()
        
        # Clear memory
        gc.collect()
        
        # Start memory tracking
        tracemalloc.start()
        
        # Measure execution time
        start_time = time.time()
        
        # Execute the function with each set of arguments
        semaphore = asyncio.Semaphore(concurrency)
        
        async def execute_with_semaphore(args, kwargs):
            async with semaphore:
                return await func(*args, **kwargs)
        
        tasks = []
        for args, kwargs in zip(args_list, kwargs_list):
            tasks.append(execute_with_semaphore(args, kwargs))
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Measure memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate throughput (operations per second)
        throughput = len(args_list) / execution_time
        
        # Add results
        result.add_execution_time(execution_time)
        result.add_memory_usage(peak / 1024 / 1024)  # Convert to MB
        result.add_throughput(throughput)
        
        # Run teardown if provided
        if teardown:
            if asyncio.iscoroutinefunction(teardown):
                await teardown()
            else:
                teardown()
        
        # End the benchmark
        result.end()
        
        # Store the result
        self.results[name] = result
        
        return result
    
    def load_baseline(self, baseline_file: str):
        """
        Load baseline results from a file.
        
        Args:
            baseline_file: Path to the baseline file
        """
        try:
            with open(baseline_file, "r") as f:
                self.baseline_results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load baseline: {e}")
    
    def save_results(self, filename: str = None):
        """
        Save benchmark results to a file.
        
        Args:
            filename: Name of the file to save results to
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert results to dictionary
        results_dict = {
            "name": self.name,
            "timestamp": datetime.now().isoformat(),
            "results": {name: result.to_dict() for name, result in self.results.items()},
        }
        
        # Save results to file
        with open(filepath, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Saved benchmark results to {filepath}")
    
    def save_csv(self, filename: str = None):
        """
        Save benchmark results to a CSV file.
        
        Args:
            filename: Name of the file to save results to
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare CSV data
        fieldnames = [
            "name",
            "category",
            "total_time",
            "avg_execution_time",
            "min_execution_time",
            "max_execution_time",
            "avg_memory_usage",
            "avg_throughput",
        ]
        
        rows = []
        for name, result in self.results.items():
            row = {
                "name": result.name,
                "category": result.category,
                "total_time": result.total_time,
                "avg_execution_time": result.avg_execution_time,
                "min_execution_time": result.min_execution_time,
                "max_execution_time": result.max_execution_time,
                "avg_memory_usage": result.avg_memory_usage,
                "avg_throughput": result.avg_throughput,
            }
            rows.append(row)
        
        # Save results to CSV file
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Saved benchmark results to {filepath}")
    
    def plot_results(self, metric: str = "avg_execution_time", categories: List[str] = None, save_path: str = None):
        """
        Plot benchmark results.
        
        Args:
            metric: Metric to plot
            categories: Categories to include in the plot
            save_path: Path to save the plot to
        """
        # Filter results by category
        if categories:
            results = {name: result for name, result in self.results.items() if result.category in categories}
        else:
            results = self.results
        
        # Group results by category
        category_results = {}
        for name, result in results.items():
            category = result.category
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot results for each category
        bar_width = 0.35
        index = np.arange(len(category_results))
        
        for i, (category, category_result) in enumerate(category_results.items()):
            # Get metric values
            values = [getattr(result, metric) for result in category_result]
            
            # Get baseline values if available
            baseline_values = []
            for result in category_result:
                if result.name in self.baseline_results:
                    baseline_values.append(self.baseline_results[result.name].get(metric, 0))
                else:
                    baseline_values.append(0)
            
            # Plot bars
            ax.bar(index + i * bar_width, values, bar_width, label=f"{category} (Current)")
            
            if baseline_values and any(baseline_values):
                ax.bar(index + i * bar_width + bar_width / 2, baseline_values, bar_width / 2, label=f"{category} (Baseline)")
        
        # Set labels and title
        ax.set_xlabel("Category")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{self.name} - {metric.replace('_', ' ').title()}")
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(list(category_results.keys()))
        ax.legend()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
    
    def print_results(self, categories: List[str] = None):
        """
        Print benchmark results.
        
        Args:
            categories: Categories to include in the output
        """
        # Filter results by category
        if categories:
            results = {name: result for name, result in self.results.items() if result.category in categories}
        else:
            results = self.results
        
        # Print results
        print(f"\n=== {self.name} Benchmark Results ===\n")
        
        for name, result in results.items():
            print(f"--- {name} ({result.category}) ---")
            print(f"Total time: {result.total_time:.6f} seconds")
            print(f"Avg execution time: {result.avg_execution_time:.6f} seconds")
            print(f"Min execution time: {result.min_execution_time:.6f} seconds")
            print(f"Max execution time: {result.max_execution_time:.6f} seconds")
            print(f"Avg memory usage: {result.avg_memory_usage:.2f} MB")
            
            if result.throughput:
                print(f"Avg throughput: {result.avg_throughput:.2f} ops/sec")
            
            # Print comparison with baseline if available
            if name in self.baseline_results:
                baseline = self.baseline_results[name]
                
                if "avg_execution_time" in baseline:
                    baseline_time = baseline["avg_execution_time"]
                    time_diff = result.avg_execution_time - baseline_time
                    time_pct = (time_diff / baseline_time) * 100 if baseline_time else 0
                    
                    print(f"Baseline avg execution time: {baseline_time:.6f} seconds")
                    print(f"Difference: {time_diff:.6f} seconds ({time_pct:+.2f}%)")
                
                if "avg_memory_usage" in baseline:
                    baseline_memory = baseline["avg_memory_usage"]
                    memory_diff = result.avg_memory_usage - baseline_memory
                    memory_pct = (memory_diff / baseline_memory) * 100 if baseline_memory else 0
                    
                    print(f"Baseline avg memory usage: {baseline_memory:.2f} MB")
                    print(f"Difference: {memory_diff:.2f} MB ({memory_pct:+.2f}%)")
                
                if "avg_throughput" in baseline and result.throughput:
                    baseline_throughput = baseline["avg_throughput"]
                    throughput_diff = result.avg_throughput - baseline_throughput
                    throughput_pct = (throughput_diff / baseline_throughput) * 100 if baseline_throughput else 0
                    
                    print(f"Baseline avg throughput: {baseline_throughput:.2f} ops/sec")
                    print(f"Difference: {throughput_diff:.2f} ops/sec ({throughput_pct:+.2f}%)")
            
            print()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Database benchmark framework")
    parser.add_argument("--output-dir", help="Output directory for benchmark results")
    parser.add_argument("--baseline", help="Baseline file for comparison")
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = DatabaseBenchmark("DatabaseBenchmark", args.output_dir)
    
    # Load baseline if provided
    if args.baseline:
        benchmark.load_baseline(args.baseline)
    
    # Run benchmarks
    # ...
    
    # Save results
    benchmark.save_results()
    benchmark.save_csv()
    
    # Print results
    benchmark.print_results()
    
    # Plot results
    benchmark.plot_results(save_path=os.path.join(benchmark.output_dir, "execution_time.png"))
    benchmark.plot_results(metric="avg_memory_usage", save_path=os.path.join(benchmark.output_dir, "memory_usage.png"))
    benchmark.plot_results(metric="avg_throughput", save_path=os.path.join(benchmark.output_dir, "throughput.png"))


if __name__ == "__main__":
    main()