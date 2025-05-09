#!/usr/bin/env python3
"""
Performance Testing Framework for Forex Trading Platform

This script runs performance tests for the forex trading platform services
and compares the results with established baselines.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Any, Optional

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define test scenarios
SCENARIOS = {
    "normal_trading": {
        "description": "Simulates normal trading hours with moderate market activity",
        "load": 100,  # requests per second
        "duration": 1800,  # 30 minutes
        "data_file": "normal_trading_data.csv"
    },
    "market_open": {
        "description": "Simulates market open with high market activity",
        "load": 500,  # requests per second
        "duration": 900,  # 15 minutes
        "data_file": "market_open_data.csv"
    },
    "high_volatility": {
        "description": "Simulates periods of high market volatility",
        "load": 300,  # requests per second
        "duration": 1800,  # 30 minutes
        "data_file": "high_volatility_data.csv"
    },
    "overnight_processing": {
        "description": "Simulates overnight batch processing",
        "load": 50,  # requests per second
        "duration": 3600,  # 60 minutes
        "data_file": "overnight_processing_data.csv"
    }
}

# Define services
SERVICES = [
    "analysis-engine-service",
    "trading-gateway-service",
    "feature-store-service",
    "ml-integration-service",
    "strategy-execution-engine",
    "data-pipeline-service"
]

# Define endpoints for each service
SERVICE_ENDPOINTS = {
    "analysis-engine-service": [
        "/api/v1/analysis/market",
        "/api/v1/analysis/patterns",
        "/api/v1/analysis/signals",
        "/api/v1/analysis/regime"
    ],
    "trading-gateway-service": [
        "/api/v1/orders",
        "/api/v1/positions",
        "/api/v1/executions",
        "/api/v1/market-data"
    ],
    "feature-store-service": [
        "/api/v1/features",
        "/api/v1/feature-sets",
        "/api/v1/data-sources",
        "/api/v1/calculations"
    ],
    "ml-integration-service": [
        "/api/v1/models",
        "/api/v1/predictions",
        "/api/v1/training",
        "/api/v1/evaluation"
    ],
    "strategy-execution-engine": [
        "/api/v1/strategies",
        "/api/v1/backtests",
        "/api/v1/executions",
        "/api/v1/performance"
    ],
    "data-pipeline-service": [
        "/api/v1/pipelines",
        "/api/v1/data-sources",
        "/api/v1/transformations",
        "/api/v1/jobs"
    ]
}

# Define metrics to collect
METRICS = [
    "request_latency_p50",
    "request_latency_p95",
    "request_latency_p99",
    "throughput",
    "error_rate",
    "cpu_usage",
    "memory_usage",
    "disk_usage"
]

# Define baselines
def load_baselines() -> Dict[str, Any]:
    """Load performance baselines from JSON file."""
    baselines_file = os.path.join(os.path.dirname(__file__), "baselines.json")
    if not os.path.exists(baselines_file):
        return {}
    
    with open(baselines_file, "r") as f:
        return json.load(f)

def save_baselines(baselines: Dict[str, Any]) -> None:
    """Save performance baselines to JSON file."""
    baselines_file = os.path.join(os.path.dirname(__file__), "baselines.json")
    with open(baselines_file, "w") as f:
        json.dump(baselines, f, indent=2)

def run_load_test(
    service: str,
    scenario: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Run a load test for a service using a specific scenario.
    
    Args:
        service: The service to test
        scenario: The test scenario
        output_dir: Directory to store test results
        
    Returns:
        Dictionary with test results
    """
    scenario_config = SCENARIOS[scenario]
    endpoints = SERVICE_ENDPOINTS[service]
    
    print(f"Running load test for {service} using {scenario} scenario...")
    print(f"  Load: {scenario_config['load']} requests per second")
    print(f"  Duration: {scenario_config['duration']} seconds")
    print(f"  Endpoints: {', '.join(endpoints)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run load test using locust
    locust_file = os.path.join(os.path.dirname(__file__), "locustfile.py")
    csv_prefix = os.path.join(output_dir, f"{service}_{scenario}")
    
    cmd = [
        "locust",
        "--headless",
        "-f", locust_file,
        "--host", f"http://{service}:8000",
        "--users", str(scenario_config["load"]),
        "--spawn-rate", "10",
        "--run-time", f"{scenario_config['duration']}s",
        "--csv", csv_prefix,
        "--csv-full-history",
        "--only-summary"
    ]
    
    # Set environment variables for locust
    env = os.environ.copy()
    env["SERVICE"] = service
    env["ENDPOINTS"] = json.dumps(endpoints)
    env["DATA_FILE"] = scenario_config["data_file"]
    
    # Run locust
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Error running load test: {stderr}")
        return {}
    
    # Parse results
    results = {}
    
    # Load CSV files
    stats_file = f"{csv_prefix}_stats.csv"
    if os.path.exists(stats_file):
        stats_df = pd.read_csv(stats_file)
        
        # Calculate metrics
        results["request_latency_p50"] = stats_df["median_response_time"].mean()
        results["request_latency_p95"] = stats_df["95%"].mean()
        results["request_latency_p99"] = stats_df["99%"].mean()
        results["throughput"] = stats_df["requests_per_sec"].sum()
        results["error_rate"] = (stats_df["num_failures"].sum() / stats_df["num_requests"].sum()) * 100
    
    # Get resource usage metrics from Prometheus
    try:
        # CPU usage
        response = requests.get(
            "http://prometheus:9090/api/v1/query",
            params={
                "query": f'avg(system_cpu_usage_percent{{service="{service}"}})'
            }
        )
        if response.status_code == 200:
            data = response.json()
            if data["status"] == "success" and len(data["data"]["result"]) > 0:
                results["cpu_usage"] = float(data["data"]["result"][0]["value"][1])
        
        # Memory usage
        response = requests.get(
            "http://prometheus:9090/api/v1/query",
            params={
                "query": f'avg(system_memory_usage_bytes{{service="{service}"}})'
            }
        )
        if response.status_code == 200:
            data = response.json()
            if data["status"] == "success" and len(data["data"]["result"]) > 0:
                results["memory_usage"] = float(data["data"]["result"][0]["value"][1])
        
        # Disk usage
        response = requests.get(
            "http://prometheus:9090/api/v1/query",
            params={
                "query": f'avg(system_disk_usage_bytes{{service="{service}"}})'
            }
        )
        if response.status_code == 200:
            data = response.json()
            if data["status"] == "success" and len(data["data"]["result"]) > 0:
                results["disk_usage"] = float(data["data"]["result"][0]["value"][1])
    
    except Exception as e:
        print(f"Error getting resource usage metrics: {e}")
    
    return results

def compare_with_baseline(
    results: Dict[str, Any],
    service: str,
    scenario: str,
    baselines: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare test results with baselines.
    
    Args:
        results: Test results
        service: The service tested
        scenario: The test scenario
        baselines: Baseline data
        
    Returns:
        Dictionary with comparison results
    """
    if not baselines or service not in baselines or scenario not in baselines[service]:
        print(f"No baseline found for {service} - {scenario}")
        return {}
    
    baseline = baselines[service][scenario]
    comparison = {}
    
    for metric in METRICS:
        if metric in results and metric in baseline:
            comparison[metric] = {
                "current": results[metric],
                "baseline": baseline[metric],
                "diff": results[metric] - baseline[metric],
                "diff_percent": ((results[metric] - baseline[metric]) / baseline[metric]) * 100
            }
    
    return comparison

def generate_report(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: str
) -> None:
    """
    Generate a performance test report.
    
    Args:
        results: Test results for all services and scenarios
        output_dir: Directory to store the report
    """
    report_file = os.path.join(output_dir, "performance_report.md")
    
    with open(report_file, "w") as f:
        f.write("# Performance Test Report\n\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for service in results:
            f.write(f"## {service}\n\n")
            
            for scenario in results[service]:
                f.write(f"### {scenario}\n\n")
                
                if "results" in results[service][scenario]:
                    f.write("#### Test Results\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    
                    for metric, value in results[service][scenario]["results"].items():
                        if isinstance(value, float):
                            f.write(f"| {metric} | {value:.2f} |\n")
                        else:
                            f.write(f"| {metric} | {value} |\n")
                
                if "comparison" in results[service][scenario]:
                    f.write("\n#### Comparison with Baseline\n\n")
                    f.write("| Metric | Current | Baseline | Diff | Diff % |\n")
                    f.write("|--------|---------|----------|------|-------|\n")
                    
                    for metric, data in results[service][scenario]["comparison"].items():
                        current = data["current"]
                        baseline = data["baseline"]
                        diff = data["diff"]
                        diff_percent = data["diff_percent"]
                        
                        if isinstance(current, float):
                            f.write(f"| {metric} | {current:.2f} | {baseline:.2f} | {diff:.2f} | {diff_percent:.2f}% |\n")
                        else:
                            f.write(f"| {metric} | {current} | {baseline} | {diff} | {diff_percent:.2f}% |\n")
                
                f.write("\n")
    
    print(f"Report generated: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Run performance tests for forex trading platform services")
    parser.add_argument("--service", choices=SERVICES, help="Service to test")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), help="Test scenario")
    parser.add_argument("--output-dir", default="results", help="Directory to store test results")
    parser.add_argument("--update-baseline", action="store_true", help="Update baseline with test results")
    parser.add_argument("--all", action="store_true", help="Test all services and scenarios")
    
    args = parser.parse_args()
    
    # Load baselines
    baselines = load_baselines()
    
    # Determine services and scenarios to test
    services_to_test = [args.service] if args.service else SERVICES
    scenarios_to_test = [args.scenario] if args.scenario else list(SCENARIOS.keys())
    
    if not args.all and not args.service and not args.scenario:
        parser.print_help()
        return
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run tests
    results = {}
    
    for service in services_to_test:
        results[service] = {}
        
        for scenario in scenarios_to_test:
            print(f"\nTesting {service} - {scenario}...")
            
            # Run load test
            test_results = run_load_test(
                service=service,
                scenario=scenario,
                output_dir=os.path.join(output_dir, service, scenario)
            )
            
            # Compare with baseline
            comparison = compare_with_baseline(
                results=test_results,
                service=service,
                scenario=scenario,
                baselines=baselines
            )
            
            results[service][scenario] = {
                "results": test_results,
                "comparison": comparison
            }
            
            # Update baseline if requested
            if args.update_baseline:
                if service not in baselines:
                    baselines[service] = {}
                
                baselines[service][scenario] = test_results
    
    # Save updated baselines
    if args.update_baseline:
        save_baselines(baselines)
        print("\nBaselines updated.")
    
    # Generate report
    generate_report(results, output_dir)

if __name__ == "__main__":
    main()
