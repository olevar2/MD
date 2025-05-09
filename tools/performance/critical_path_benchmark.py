"""
Critical Path Performance Benchmarking Tool

This script benchmarks critical performance paths in the forex trading platform:
1. Order execution flow (strategy → execution → trading gateway)
2. Market data retrieval and processing
3. Signal generation and analysis
4. ML model inference paths

Usage:
    python critical_path_benchmark.py --path order_execution
    python critical_path_benchmark.py --path market_data
    python critical_path_benchmark.py --path signal_generation
    python critical_path_benchmark.py --path ml_inference
    python critical_path_benchmark.py --path all
"""

import os
import sys
import time
import json
import logging
import argparse
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import common library
from common_lib.monitoring.tracing import setup_tracing, trace_async_function
from common_lib.monitoring.metrics import setup_metrics, track_execution_time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "reports" / "performance"
DEFAULT_ITERATIONS = 10
DEFAULT_LOAD_LEVELS = ["low", "medium", "high"]
DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
DEFAULT_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

class CriticalPathBenchmark:
    """Benchmark critical performance paths in the forex trading platform."""
    
    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        iterations: int = DEFAULT_ITERATIONS,
        load_levels: List[str] = None,
        symbols: List[str] = None,
        timeframes: List[str] = None
    ):
        """
        Initialize the benchmark.
        
        Args:
            output_dir: Directory to store benchmark results
            iterations: Number of iterations for each benchmark
            load_levels: Load levels to test
            symbols: Currency symbols to use in tests
            timeframes: Timeframes to use in tests
        """
        self.output_dir = output_dir
        self.iterations = iterations
        self.load_levels = load_levels or DEFAULT_LOAD_LEVELS
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.timeframes = timeframes or DEFAULT_TIMEFRAMES
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        self.results = {}
        
        # Set up tracing
        setup_tracing(service_name="benchmark-tool")
        
        # Set up metrics
        setup_metrics(service_name="benchmark-tool")
        
        logger.info(f"Benchmark initialized with {iterations} iterations")
        logger.info(f"Output directory: {output_dir}")
    
    async def benchmark_order_execution(self) -> Dict[str, Any]:
        """
        Benchmark order execution flow.
        
        Returns:
            Benchmark results
        """
        logger.info("Benchmarking order execution flow")
        
        results = {
            "path": "order_execution",
            "iterations": self.iterations,
            "load_levels": {},
            "bottlenecks": []
        }
        
        for load_level in self.load_levels:
            logger.info(f"Testing load level: {load_level}")
            
            # Determine number of concurrent orders based on load level
            concurrent_orders = {
                "low": 1,
                "medium": 10,
                "high": 50
            }.get(load_level, 1)
            
            # Run benchmark
            start_time = time.time()
            execution_times = []
            
            for i in range(self.iterations):
                iteration_start = time.time()
                
                # Simulate order execution flow
                await self._simulate_order_execution(concurrent_orders)
                
                iteration_time = time.time() - iteration_start
                execution_times.append(iteration_time)
                
                logger.info(f"Iteration {i+1}/{self.iterations}: {iteration_time:.4f}s")
            
            # Calculate statistics
            total_time = time.time() - start_time
            avg_time = sum(execution_times) / len(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)
            
            results["load_levels"][load_level] = {
                "total_time": total_time,
                "avg_time": avg_time,
                "max_time": max_time,
                "min_time": min_time,
                "execution_times": execution_times
            }
            
            logger.info(f"Load level {load_level} completed: avg={avg_time:.4f}s, max={max_time:.4f}s, min={min_time:.4f}s")
        
        # Identify bottlenecks
        results["bottlenecks"] = self._analyze_bottlenecks("order_execution")
        
        # Save results
        self._save_results("order_execution", results)
        
        return results
    
    async def benchmark_market_data(self) -> Dict[str, Any]:
        """
        Benchmark market data retrieval and processing.
        
        Returns:
            Benchmark results
        """
        logger.info("Benchmarking market data retrieval and processing")
        
        results = {
            "path": "market_data",
            "iterations": self.iterations,
            "load_levels": {},
            "bottlenecks": []
        }
        
        for load_level in self.load_levels:
            logger.info(f"Testing load level: {load_level}")
            
            # Determine data size based on load level
            data_points = {
                "low": 1000,
                "medium": 10000,
                "high": 100000
            }.get(load_level, 1000)
            
            # Run benchmark
            start_time = time.time()
            execution_times = []
            
            for i in range(self.iterations):
                iteration_start = time.time()
                
                # Simulate market data retrieval and processing
                await self._simulate_market_data_processing(data_points)
                
                iteration_time = time.time() - iteration_start
                execution_times.append(iteration_time)
                
                logger.info(f"Iteration {i+1}/{self.iterations}: {iteration_time:.4f}s")
            
            # Calculate statistics
            total_time = time.time() - start_time
            avg_time = sum(execution_times) / len(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)
            
            results["load_levels"][load_level] = {
                "total_time": total_time,
                "avg_time": avg_time,
                "max_time": max_time,
                "min_time": min_time,
                "execution_times": execution_times
            }
            
            logger.info(f"Load level {load_level} completed: avg={avg_time:.4f}s, max={max_time:.4f}s, min={min_time:.4f}s")
        
        # Identify bottlenecks
        results["bottlenecks"] = self._analyze_bottlenecks("market_data")
        
        # Save results
        self._save_results("market_data", results)
        
        return results
    
    async def benchmark_signal_generation(self) -> Dict[str, Any]:
        """
        Benchmark signal generation and analysis.
        
        Returns:
            Benchmark results
        """
        logger.info("Benchmarking signal generation and analysis")
        
        results = {
            "path": "signal_generation",
            "iterations": self.iterations,
            "load_levels": {},
            "bottlenecks": []
        }
        
        for load_level in self.load_levels:
            logger.info(f"Testing load level: {load_level}")
            
            # Determine complexity based on load level
            indicators_count = {
                "low": 5,
                "medium": 20,
                "high": 50
            }.get(load_level, 5)
            
            # Run benchmark
            start_time = time.time()
            execution_times = []
            
            for i in range(self.iterations):
                iteration_start = time.time()
                
                # Simulate signal generation and analysis
                await self._simulate_signal_generation(indicators_count)
                
                iteration_time = time.time() - iteration_start
                execution_times.append(iteration_time)
                
                logger.info(f"Iteration {i+1}/{self.iterations}: {iteration_time:.4f}s")
            
            # Calculate statistics
            total_time = time.time() - start_time
            avg_time = sum(execution_times) / len(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)
            
            results["load_levels"][load_level] = {
                "total_time": total_time,
                "avg_time": avg_time,
                "max_time": max_time,
                "min_time": min_time,
                "execution_times": execution_times
            }
            
            logger.info(f"Load level {load_level} completed: avg={avg_time:.4f}s, max={max_time:.4f}s, min={min_time:.4f}s")
        
        # Identify bottlenecks
        results["bottlenecks"] = self._analyze_bottlenecks("signal_generation")
        
        # Save results
        self._save_results("signal_generation", results)
        
        return results
    
    async def benchmark_ml_inference(self) -> Dict[str, Any]:
        """
        Benchmark ML model inference paths.
        
        Returns:
            Benchmark results
        """
        logger.info("Benchmarking ML model inference paths")
        
        results = {
            "path": "ml_inference",
            "iterations": self.iterations,
            "load_levels": {},
            "bottlenecks": []
        }
        
        for load_level in self.load_levels:
            logger.info(f"Testing load level: {load_level}")
            
            # Determine model complexity based on load level
            model_complexity = {
                "low": "simple",
                "medium": "medium",
                "high": "complex"
            }.get(load_level, "simple")
            
            # Run benchmark
            start_time = time.time()
            execution_times = []
            
            for i in range(self.iterations):
                iteration_start = time.time()
                
                # Simulate ML model inference
                await self._simulate_ml_inference(model_complexity)
                
                iteration_time = time.time() - iteration_start
                execution_times.append(iteration_time)
                
                logger.info(f"Iteration {i+1}/{self.iterations}: {iteration_time:.4f}s")
            
            # Calculate statistics
            total_time = time.time() - start_time
            avg_time = sum(execution_times) / len(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)
            
            results["load_levels"][load_level] = {
                "total_time": total_time,
                "avg_time": avg_time,
                "max_time": max_time,
                "min_time": min_time,
                "execution_times": execution_times
            }
            
            logger.info(f"Load level {load_level} completed: avg={avg_time:.4f}s, max={max_time:.4f}s, min={min_time:.4f}s")
        
        # Identify bottlenecks
        results["bottlenecks"] = self._analyze_bottlenecks("ml_inference")
        
        # Save results
        self._save_results("ml_inference", results)
        
        return results
