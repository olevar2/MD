#!/usr/bin/env python3
"""
Performance Metrics Analyzer

This script analyzes performance metrics of the forex trading platform:
1. Response times for critical API endpoints
2. Database query execution times
3. Memory usage patterns
4. CPU utilization during peak loads

Output is a JSON file with comprehensive performance metrics.
"""

import os
import sys
import json
import time
import logging
import argparse
import statistics
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import concurrent.futures
import psutil
import sqlite3
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"
DEFAULT_OUTPUT_DIR = r"D:\MD\forex_trading_platform\tools\output"
DEFAULT_LOG_DIR = r"D:\MD\forex_trading_platform\logs"

class PerformanceMetricsAnalyzer:
    """Analyzes performance metrics of the forex trading platform."""
    
    def __init__(self, project_root: Path, log_dir: Path):
        """
        Initialize the analyzer.
        
        Args:
            project_root: Root directory of the project
            log_dir: Directory containing log files
        """
        self.project_root = project_root
        self.log_dir = log_dir
        self.api_response_times = {}
        self.db_query_times = {}
        self.memory_usage = {}
        self.cpu_usage = {}
        
    def analyze_api_response_times(self) -> Dict[str, Any]:
        """
        Analyze API response times from log files.
        
        Returns:
            Dictionary with API response time metrics
        """
        logger.info("Analyzing API response times...")
        
        response_times = defaultdict(list)
        
        # Look for log files
        log_files = []
        for root, _, files in os.walk(self.log_dir):
            for file in files:
                if file.endswith('.log') and ('api' in file.lower() or 'service' in file.lower()):
                    log_files.append(os.path.join(root, file))
        
        # Extract response times from logs
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Look for response time patterns
                        # Example: "2023-05-10 12:34:56 INFO Request to /api/v1/prices completed in 123.45ms"
                        match = re.search(r'Request to (/[^\s]+) completed in (\d+\.?\d*)ms', line)
                        if match:
                            endpoint = match.group(1)
                            response_time = float(match.group(2))
                            response_times[endpoint].append(response_time)
            except Exception as e:
                logger.error(f"Error processing log file {log_file}: {e}")
        
        # Calculate statistics
        results = {}
        for endpoint, times in response_times.items():
            if times:
                results[endpoint] = {
                    'count': len(times),
                    'min': min(times),
                    'max': max(times),
                    'avg': statistics.mean(times),
                    'median': statistics.median(times),
                    'p95': sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else None,
                    'p99': sorted(times)[int(len(times) * 0.99)] if len(times) >= 100 else None
                }
        
        return {
            'api_response_times': results,
            'total_endpoints': len(results),
            'total_requests': sum(len(times) for times in response_times.values())
        }
    
    def analyze_db_query_times(self) -> Dict[str, Any]:
        """
        Analyze database query execution times.
        
        Returns:
            Dictionary with database query time metrics
        """
        logger.info("Analyzing database query times...")
        
        query_times = defaultdict(list)
        
        # Look for log files
        log_files = []
        for root, _, files in os.walk(self.log_dir):
            for file in files:
                if file.endswith('.log') and ('db' in file.lower() or 'sql' in file.lower() or 'database' in file.lower()):
                    log_files.append(os.path.join(root, file))
        
        # Extract query times from logs
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Look for query time patterns
                        # Example: "2023-05-10 12:34:56 INFO Query 'SELECT * FROM prices' executed in 45.67ms"
                        match = re.search(r'Query [\'"]([^\'"]*)[\'"]\s+executed in (\d+\.?\d*)ms', line)
                        if match:
                            query = match.group(1)
                            # Simplify query by removing specific values
                            simplified_query = re.sub(r'\'[^\']*\'', '?', query)
                            simplified_query = re.sub(r'\d+', '?', simplified_query)
                            execution_time = float(match.group(2))
                            query_times[simplified_query].append(execution_time)
            except Exception as e:
                logger.error(f"Error processing log file {log_file}: {e}")
        
        # Calculate statistics
        results = {}
        for query, times in query_times.items():
            if times:
                results[query] = {
                    'count': len(times),
                    'min': min(times),
                    'max': max(times),
                    'avg': statistics.mean(times),
                    'median': statistics.median(times),
                    'p95': sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else None,
                    'p99': sorted(times)[int(len(times) * 0.99)] if len(times) >= 100 else None
                }
        
        # Identify slow queries
        slow_queries = {query: stats for query, stats in results.items() if stats['avg'] > 100}  # Queries taking more than 100ms on average
        
        return {
            'db_query_times': results,
            'total_queries': len(results),
            'total_executions': sum(len(times) for times in query_times.values()),
            'slow_queries': slow_queries,
            'slow_query_count': len(slow_queries)
        }
    
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """
        Analyze memory usage patterns.
        
        Returns:
            Dictionary with memory usage metrics
        """
        logger.info("Analyzing memory usage patterns...")
        
        memory_usage = defaultdict(list)
        
        # Look for log files
        log_files = []
        for root, _, files in os.walk(self.log_dir):
            for file in files:
                if file.endswith('.log') and ('memory' in file.lower() or 'performance' in file.lower() or 'monitoring' in file.lower()):
                    log_files.append(os.path.join(root, file))
        
        # Extract memory usage from logs
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Look for memory usage patterns
                        # Example: "2023-05-10 12:34:56 INFO Service 'trading-gateway' memory usage: 256.78 MB"
                        match = re.search(r'Service [\'"]([^\'"]*)[\'"] memory usage: (\d+\.?\d*) MB', line)
                        if match:
                            service = match.group(1)
                            memory_mb = float(match.group(2))
                            memory_usage[service].append(memory_mb)
            except Exception as e:
                logger.error(f"Error processing log file {log_file}: {e}")
        
        # Calculate statistics
        results = {}
        for service, usages in memory_usage.items():
            if usages:
                results[service] = {
                    'count': len(usages),
                    'min': min(usages),
                    'max': max(usages),
                    'avg': statistics.mean(usages),
                    'median': statistics.median(usages),
                    'trend': 'increasing' if usages[-1] > usages[0] else 'decreasing' if usages[-1] < usages[0] else 'stable'
                }
        
        # Identify high memory services
        high_memory_services = {service: stats for service, stats in results.items() if stats['avg'] > 500}  # Services using more than 500MB on average
        
        return {
            'memory_usage': results,
            'total_services': len(results),
            'high_memory_services': high_memory_services,
            'high_memory_service_count': len(high_memory_services)
        }
    
    def analyze_cpu_usage(self) -> Dict[str, Any]:
        """
        Analyze CPU utilization during peak loads.
        
        Returns:
            Dictionary with CPU usage metrics
        """
        logger.info("Analyzing CPU utilization...")
        
        cpu_usage = defaultdict(list)
        
        # Look for log files
        log_files = []
        for root, _, files in os.walk(self.log_dir):
            for file in files:
                if file.endswith('.log') and ('cpu' in file.lower() or 'performance' in file.lower() or 'monitoring' in file.lower()):
                    log_files.append(os.path.join(root, file))
        
        # Extract CPU usage from logs
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Look for CPU usage patterns
                        # Example: "2023-05-10 12:34:56 INFO Service 'trading-gateway' CPU usage: 45.67%"
                        match = re.search(r'Service [\'"]([^\'"]*)[\'"] CPU usage: (\d+\.?\d*)%', line)
                        if match:
                            service = match.group(1)
                            cpu_percent = float(match.group(2))
                            cpu_usage[service].append(cpu_percent)
            except Exception as e:
                logger.error(f"Error processing log file {log_file}: {e}")
        
        # Calculate statistics
        results = {}
        for service, usages in cpu_usage.items():
            if usages:
                results[service] = {
                    'count': len(usages),
                    'min': min(usages),
                    'max': max(usages),
                    'avg': statistics.mean(usages),
                    'median': statistics.median(usages),
                    'peak_periods': sum(1 for usage in usages if usage > 80)  # Count periods with >80% CPU usage
                }
        
        # Identify high CPU services
        high_cpu_services = {service: stats for service, stats in results.items() if stats['avg'] > 50}  # Services using more than 50% CPU on average
        
        return {
            'cpu_usage': results,
            'total_services': len(results),
            'high_cpu_services': high_cpu_services,
            'high_cpu_service_count': len(high_cpu_services)
        }
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        # Analyze API response times
        api_metrics = self.analyze_api_response_times()
        
        # Analyze database query times
        db_metrics = self.analyze_db_query_times()
        
        # Analyze memory usage
        memory_metrics = self.analyze_memory_usage()
        
        # Analyze CPU usage
        cpu_metrics = self.analyze_cpu_usage()
        
        # Combine results
        return {
            'timestamp': datetime.now().isoformat(),
            'api_metrics': api_metrics,
            'db_metrics': db_metrics,
            'memory_metrics': memory_metrics,
            'cpu_metrics': cpu_metrics,
            'summary': {
                'total_endpoints': api_metrics.get('total_endpoints', 0),
                'total_queries': db_metrics.get('total_queries', 0),
                'slow_query_count': db_metrics.get('slow_query_count', 0),
                'high_memory_service_count': memory_metrics.get('high_memory_service_count', 0),
                'high_cpu_service_count': cpu_metrics.get('high_cpu_service_count', 0)
            }
        }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze performance metrics")
    parser.add_argument(
        "--project-root", 
        default=DEFAULT_PROJECT_ROOT,
        help="Root directory of the project"
    )
    parser.add_argument(
        "--output-dir", 
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--log-dir", 
        default=DEFAULT_LOG_DIR,
        help="Directory containing log files"
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze performance metrics
    analyzer = PerformanceMetricsAnalyzer(Path(args.project_root), Path(args.log_dir))
    results = analyzer.analyze()
    
    # Save results
    output_path = os.path.join(args.output_dir, "performance_metrics.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Performance metrics saved to {output_path}")

if __name__ == "__main__":
    from collections import defaultdict
    main()
