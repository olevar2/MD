#!/usr/bin/env python
"""
Script to run all database benchmarks.

This script runs all database benchmarks and generates a comprehensive report.
"""
import os
import sys
import argparse
import subprocess
import logging
import time
import json
import asyncio
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run database benchmarks")
    parser.add_argument(
        "--db-host",
        default="localhost",
        help="Database host",
    )
    parser.add_argument(
        "--db-port",
        type=int,
        default=5432,
        help="Database port",
    )
    parser.add_argument(
        "--db-user",
        default="postgres",
        help="Database user",
    )
    parser.add_argument(
        "--db-password",
        default="postgres",
        help="Database password",
    )
    parser.add_argument(
        "--db-name",
        default="forex_platform",
        help="Database name",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Use Docker for test environment",
    )
    parser.add_argument(
        "--benchmark",
        choices=["connection_pool", "prepared_statements", "bulk_operations", "all"],
        default="all",
        help="Benchmark to run",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/database/results",
        help="Output directory for benchmark results",
    )
    parser.add_argument(
        "--baseline",
        help="Baseline file for comparison",
    )
    return parser.parse_args()


def setup_docker_environment():
    """Set up Docker environment for benchmarking."""
    logger.info("Setting up Docker environment...")
    
    # Check if Docker is installed
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Docker is not installed or not in PATH")
        sys.exit(1)
    
    # Check if Docker Compose is installed
    try:
        subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Docker Compose is not installed or not in PATH")
        sys.exit(1)
    
    # Create a docker-compose.yml file for benchmarking
    with open("docker-compose.benchmark.yml", "w") as f:
        f.write("""
version: '3'

services:
  postgres:
    image: timescale/timescaledb:latest-pg13
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: forex_platform
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
""")
    
    # Start the Docker containers
    subprocess.run(
        ["docker-compose", "-f", "docker-compose.benchmark.yml", "up", "-d"],
        check=True,
    )
    
    # Wait for the containers to start
    logger.info("Waiting for containers to start...")
    time.sleep(10)
    
    # Check if the containers are running
    result = subprocess.run(
        ["docker-compose", "-f", "docker-compose.benchmark.yml", "ps"],
        check=True,
        capture_output=True,
        text=True,
    )
    
    if "Up" not in result.stdout:
        logger.error("Containers failed to start")
        sys.exit(1)
    
    logger.info("Docker environment is ready")


def teardown_docker_environment():
    """Tear down Docker environment."""
    logger.info("Tearing down Docker environment...")
    
    # Stop the Docker containers
    subprocess.run(
        ["docker-compose", "-f", "docker-compose.benchmark.yml", "down", "-v"],
        check=True,
    )
    
    # Remove the docker-compose.yml file
    os.remove("docker-compose.benchmark.yml")
    
    logger.info("Docker environment has been torn down")


def run_benchmark(benchmark, args):
    """
    Run a specific benchmark.
    
    Args:
        benchmark: Name of the benchmark to run
        args: Command line arguments
    """
    logger.info(f"Running {benchmark} benchmark...")
    
    # Set environment variables for the benchmark
    env = os.environ.copy()
    env["DB_HOST"] = args.db_host
    env["DB_PORT"] = str(args.db_port)
    env["DB_USER"] = args.db_user
    env["DB_PASSWORD"] = args.db_password
    env["DB_NAME"] = args.db_name
    
    # Run the benchmark
    benchmark_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"benchmark_{benchmark}.py")
    subprocess.run(
        [sys.executable, benchmark_script],
        env=env,
        check=True,
    )
    
    logger.info(f"{benchmark} benchmark complete")


def generate_report(args):
    """
    Generate a comprehensive benchmark report.
    
    Args:
        args: Command line arguments
    """
    logger.info("Generating benchmark report...")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all benchmark result files
    result_files = []
    for root, dirs, files in os.walk(args.output_dir):
        for file in files:
            if file.endswith(".json") and not file.endswith("_report.json"):
                result_files.append(os.path.join(root, file))
    
    # Load benchmark results
    results = {}
    for file in result_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                
                # Add results to combined results
                for name, result in data.get("results", {}).items():
                    results[name] = result
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load results from {file}: {e}")
    
    # Group results by category
    categories = {}
    for name, result in results.items():
        category = result.get("category", "unknown")
        if category not in categories:
            categories[category] = []
        categories[category].append(result)
    
    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "categories": {},
        "summary": {},
    }
    
    # Add category results
    for category, category_results in categories.items():
        report["categories"][category] = {
            "count": len(category_results),
            "results": category_results,
        }
    
    # Add summary
    report["summary"] = {
        "total_benchmarks": len(results),
        "categories": list(categories.keys()),
        "category_counts": {category: len(results) for category, results in categories.items()},
    }
    
    # Save report
    report_file = os.path.join(args.output_dir, "benchmark_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    # Generate HTML report
    html_report = os.path.join(args.output_dir, "benchmark_report.html")
    with open(html_report, "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Database Benchmark Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
        }
        h1, h2, h3 {
            color: #333;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .chart {
            width: 100%;
            height: 400px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <h1>Database Benchmark Report</h1>
    <p>Generated: """ + report["timestamp"] + """</p>
    
    <h2>Summary</h2>
    <p>Total benchmarks: """ + str(report["summary"]["total_benchmarks"]) + """</p>
    
    <h3>Categories</h3>
    <ul>
""")
        
        for category, count in report["summary"]["category_counts"].items():
            f.write(f"        <li>{category}: {count} benchmarks</li>\n")
        
        f.write("""
    </ul>
    
    <h2>Results by Category</h2>
""")
        
        for category, category_data in report["categories"].items():
            f.write(f"""
    <h3>{category}</h3>
    <table>
        <tr>
            <th>Name</th>
            <th>Avg Execution Time (s)</th>
            <th>Avg Memory Usage (MB)</th>
            <th>Avg Throughput (ops/sec)</th>
        </tr>
""")
            
            for result in category_data["results"]:
                f.write(f"""
        <tr>
            <td>{result["name"]}</td>
            <td>{result.get("avg_execution_time", 0):.6f}</td>
            <td>{result.get("avg_memory_usage", 0):.2f}</td>
            <td>{result.get("avg_throughput", 0):.2f}</td>
        </tr>
""")
            
            f.write("""
    </table>
""")
        
        f.write("""
    <h2>Charts</h2>
    
    <h3>Execution Time by Category</h3>
    <img class="chart" src="execution_time_by_category.png" alt="Execution Time by Category">
    
    <h3>Memory Usage by Category</h3>
    <img class="chart" src="memory_usage_by_category.png" alt="Memory Usage by Category">
    
    <h3>Throughput by Category</h3>
    <img class="chart" src="throughput_by_category.png" alt="Throughput by Category">
</body>
</html>
""")
    
    # Generate charts
    generate_charts(report, args.output_dir)
    
    logger.info(f"Benchmark report generated: {html_report}")


def generate_charts(report, output_dir):
    """
    Generate charts for the benchmark report.
    
    Args:
        report: Benchmark report data
        output_dir: Output directory for charts
    """
    # Generate execution time chart
    plt.figure(figsize=(12, 8))
    
    categories = []
    avg_times = []
    
    for category, category_data in report["categories"].items():
        if category == "unknown":
            continue
        
        categories.append(category)
        avg_time = sum(result.get("avg_execution_time", 0) for result in category_data["results"]) / len(category_data["results"])
        avg_times.append(avg_time)
    
    plt.bar(categories, avg_times)
    plt.xlabel("Category")
    plt.ylabel("Average Execution Time (s)")
    plt.title("Average Execution Time by Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "execution_time_by_category.png"))
    
    # Generate memory usage chart
    plt.figure(figsize=(12, 8))
    
    categories = []
    avg_memory = []
    
    for category, category_data in report["categories"].items():
        if category == "unknown":
            continue
        
        categories.append(category)
        avg_mem = sum(result.get("avg_memory_usage", 0) for result in category_data["results"]) / len(category_data["results"])
        avg_memory.append(avg_mem)
    
    plt.bar(categories, avg_memory)
    plt.xlabel("Category")
    plt.ylabel("Average Memory Usage (MB)")
    plt.title("Average Memory Usage by Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "memory_usage_by_category.png"))
    
    # Generate throughput chart
    plt.figure(figsize=(12, 8))
    
    categories = []
    avg_throughput = []
    
    for category, category_data in report["categories"].items():
        if category == "unknown":
            continue
        
        # Skip categories without throughput data
        throughputs = [result.get("avg_throughput", 0) for result in category_data["results"]]
        if not any(throughputs):
            continue
        
        categories.append(category)
        avg_thru = sum(throughputs) / len(throughputs)
        avg_throughput.append(avg_thru)
    
    plt.bar(categories, avg_throughput)
    plt.xlabel("Category")
    plt.ylabel("Average Throughput (ops/sec)")
    plt.title("Average Throughput by Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "throughput_by_category.png"))


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Set up Docker environment if requested
        if args.docker:
            setup_docker_environment()
        
        # Run benchmarks
        if args.benchmark == "all" or args.benchmark == "connection_pool":
            run_benchmark("connection_pool", args)
        
        if args.benchmark == "all" or args.benchmark == "prepared_statements":
            run_benchmark("prepared_statements", args)
        
        if args.benchmark == "all" or args.benchmark == "bulk_operations":
            run_benchmark("bulk_operations", args)
        
        # Generate report
        generate_report(args)
    finally:
        # Tear down Docker environment if it was set up
        if args.docker:
            teardown_docker_environment()


if __name__ == "__main__":
    main()