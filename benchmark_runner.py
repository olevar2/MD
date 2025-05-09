"""
Comprehensive Performance Benchmark Runner

This script combines and runs all available performance benchmarks 
for the forex trading platform.
"""

import logging
import time
import os
import sys
import gc
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import psutil
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveBenchmark:
    """Run comprehensive performance benchmarks across all components."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.path.join("testing", "benchmark_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize results containers
        self.results = {
            'indicators': [],
            'memory': [],
            'execution_time': [],
            'accuracy': []
        }
        
    def generate_test_data(self, size: int = 10000) -> pd.DataFrame:
        """Generate sample OHLCV data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=size, freq='5min')
        prices = np.random.random(size) * 100 + 50  # Base price around 100
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.normal(0, 1, size),
            'high': prices + np.random.normal(2, 1, size),
            'low': prices + np.random.normal(-2, 1, size),
            'close': prices + np.random.normal(0, 1, size),
            'volume': np.random.randint(1000, 10000, size)
        }).set_index('timestamp')

    def measure_execution_time(self, func, *args, repeats=5) -> float:
        """Measure execution time of a function."""
        times = []
        for _ in range(repeats):
            start_time = time.time()
            func(*args)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        return sum(times) / len(times)

    def measure_memory_usage(self, func, *args) -> float:
        """Measure memory usage of a function."""
        gc.collect()
        process = psutil.Process()
        
        # Get baseline
        baseline = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Run function
        result = func(*args)
        
        # Get peak
        peak = process.memory_info().rss / (1024 * 1024)
        
        # Calculate difference
        memory_used = peak - baseline
        
        return memory_used

    def run_benchmarks(self):
        """Run all benchmarks."""
        logger.info("Starting comprehensive benchmarks...")
        
        # Generate test data of different sizes
        sizes = [1000, 10000, 100000]
        for size in sizes:
            logger.info(f"Testing with data size: {size}")
            data = self.generate_test_data(size)
            
            # Example benchmark function
            def test_function(df):
                # Simulate some calculations
                return df.rolling(20).mean()
            
            # Measure performance
            execution_time = self.measure_execution_time(test_function, data)
            memory_usage = self.measure_memory_usage(test_function, data)
            
            # Store results
            self.results['execution_time'].append({
                'size': size,
                'time_ms': execution_time
            })
            
            self.results['memory'].append({
                'size': size,
                'memory_mb': memory_usage
            })
            
            logger.info(f"Size {size}: Time {execution_time:.2f}ms, Memory {memory_usage:.2f}MB")
        
        self.save_results()
        self.generate_report()

    def save_results(self):
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.json")
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"Results saved to {filepath}")

    def generate_report(self):
        """Generate HTML report with plots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"benchmark_report_{timestamp}.html")
        
        # Create plots
        plt.figure(figsize=(10, 6))
        
        # Execution time plot
        sizes = [r['size'] for r in self.results['execution_time']]
        times = [r['time_ms'] for r in self.results['execution_time']]
        plt.subplot(2, 1, 1)
        plt.plot(sizes, times, 'b-o')
        plt.title('Execution Time vs Data Size')
        plt.xlabel('Data Size')
        plt.ylabel('Time (ms)')
        
        # Memory usage plot
        memory = [r['memory_mb'] for r in self.results['memory']]
        plt.subplot(2, 1, 2)
        plt.plot(sizes, memory, 'r-o')
        plt.title('Memory Usage vs Data Size')
        plt.xlabel('Data Size')
        plt.ylabel('Memory (MB)')
        
        plt.tight_layout()
        
        # Save plots
        plot_path = os.path.join(self.output_dir, f"benchmark_plots_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Generate HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Forex Trading Platform Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Performance Benchmark Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Performance Summary</h2>
            <table>
                <tr>
                    <th>Data Size</th>
                    <th>Execution Time (ms)</th>
                    <th>Memory Usage (MB)</th>
                </tr>
        """
        
        # Add rows for each result
        for i in range(len(sizes)):
            html += f"""
                <tr>
                    <td>{sizes[i]}</td>
                    <td>{times[i]:.2f}</td>
                    <td>{memory[i]:.2f}</td>
                </tr>
            """
            
        html += f"""
            </table>
            
            <h2>Performance Plots</h2>
            <img src="{os.path.basename(plot_path)}" alt="Performance Plots">
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html)
            
        logger.info(f"Report generated at {filepath}")

def main():
    benchmark = ComprehensiveBenchmark()
    benchmark.run_benchmarks()

if __name__ == "__main__":
    main()
