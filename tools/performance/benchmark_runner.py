"""
Benchmark Runner for Critical Path Performance Testing

This script runs the critical path benchmarks and generates reports.

Usage:
    python benchmark_runner.py --path all
    python benchmark_runner.py --path order_execution
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import benchmark class
from critical_path_benchmark import CriticalPathBenchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "reports" / "performance"
DEFAULT_ITERATIONS = 10

async def run_benchmarks(args):
    """Run benchmarks based on command line arguments."""
    benchmark = CriticalPathBenchmark(
        output_dir=Path(args.output_dir),
        iterations=args.iterations
    )
    
    if args.path == "order_execution" or args.path == "all":
        await benchmark.benchmark_order_execution()
    
    if args.path == "market_data" or args.path == "all":
        await benchmark.benchmark_market_data()
    
    if args.path == "signal_generation" or args.path == "all":
        await benchmark.benchmark_signal_generation()
    
    if args.path == "ml_inference" or args.path == "all":
        await benchmark.benchmark_ml_inference()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark critical performance paths")
    parser.add_argument(
        "--path",
        choices=["order_execution", "market_data", "signal_generation", "ml_inference", "all"],
        default="all",
        help="Path to benchmark"
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store benchmark results"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Number of iterations for each benchmark"
    )
    
    args = parser.parse_args()
    
    # Run benchmarks
    asyncio.run(run_benchmarks(args))

if __name__ == "__main__":
    main()
