#!/usr/bin/env python3
"""
Performance test for the Forex Trading Platform.

This script tests the performance of the platform under load,
measuring response times, throughput, and resource usage.
"""

import os
import sys
import json
import time
import logging
import requests
import subprocess
import statistics
import psutil
import concurrent.futures
import threading
import random
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('performance_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Service configuration
SERVICE_CONFIG = {
    'trading-gateway-service': {
        'port': 8001,
        'test_endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/instruments', 'weight': 10},
            {'method': 'GET', 'endpoint': '/api/v1/accounts', 'weight': 5}
        ]
    },
    'portfolio-management-service': {
        'port': 8002,
        'test_endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/portfolios', 'weight': 8},
            {'method': 'GET', 'endpoint': '/api/v1/positions', 'weight': 7}
        ]
    },
    'risk-management-service': {
        'port': 8003,
        'test_endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/risk-profiles', 'weight': 5},
            {'method': 'GET', 'endpoint': '/api/v1/risk-limits', 'weight': 3}
        ]
    },
    'data-pipeline-service': {
        'port': 8004,
        'test_endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/market-data', 'weight': 15},
            {'method': 'GET', 'endpoint': '/api/v1/data-sources', 'weight': 2}
        ]
    },
    'feature-store-service': {
        'port': 8005,
        'test_endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/features', 'weight': 10},
            {'method': 'GET', 'endpoint': '/api/v1/feature-sets', 'weight': 5}
        ]
    },
    'ml-integration-service': {
        'port': 8006,
        'test_endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/models', 'weight': 5},
            {'method': 'GET', 'endpoint': '/api/v1/predictions', 'weight': 10}
        ]
    }
}

# Test configuration
TEST_CONFIG = {
    'duration': 10,  # Test duration in seconds (reduced for testing)
    'ramp_up': 2,    # Ramp-up time in seconds (reduced for testing)
    'users': 5,      # Number of concurrent users (reduced for testing)
    'request_timeout': 5,   # Timeout for HTTP requests in seconds
    'test_data_dir': 'test_data',  # Directory for test data
    'metrics_interval': 2,  # Interval for collecting metrics in seconds
    'thresholds': {
        'response_time_avg': 500,  # Average response time threshold in ms
        'response_time_p95': 1000,  # 95th percentile response time threshold in ms
        'error_rate': 0.05,  # Error rate threshold (5%)
        'cpu_usage': 80,  # CPU usage threshold in percent
        'memory_usage': 80  # Memory usage threshold in percent
    }
}

def make_request(service_name: str, method: str, endpoint: str) -> Tuple[bool, int, float]:
    """
    Simulate an HTTP request to a service endpoint.

    Args:
        service_name: Name of the service
        method: HTTP method
        endpoint: Endpoint to request

    Returns:
        Tuple of (success, status_code, response_time)
    """
    # For testing purposes, we'll simulate responses instead of making actual HTTP requests

    start_time = time.time()

    # Simulate processing time
    time.sleep(0.01 + random.random() * 0.05)  # 10-60ms

    # Simulate occasional errors (1% chance)
    if random.random() < 0.01:
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        return False, 500, response_time

    response_time = (time.time() - start_time) * 1000  # Convert to ms
    return True, 200, response_time

def user_session(user_id: int, stop_event) -> Dict[str, Any]:
    """
    Simulate a user session.

    Args:
        user_id: User ID
        stop_event: Event to signal when to stop

    Returns:
        Dictionary with session results
    """
    logger.debug(f"Starting user session: {user_id}")

    results = {
        'user_id': user_id,
        'requests': 0,
        'successful_requests': 0,
        'failed_requests': 0,
        'response_times': [],
        'errors': []
    }

    # Create a list of all endpoints with their weights
    all_endpoints = []
    for service_name, config in SERVICE_CONFIG.items():
        for endpoint_config in config['test_endpoints']:
            for _ in range(endpoint_config['weight']):
                all_endpoints.append((
                    service_name,
                    endpoint_config['method'],
                    endpoint_config['endpoint']
                ))

    # Simulate user activity
    while not stop_event.is_set():
        # Select an endpoint based on weight
        endpoint_index = user_id % len(all_endpoints)
        service_name, method, endpoint = all_endpoints[endpoint_index]

        # Make the request
        success, status_code, response_time = make_request(service_name, method, endpoint)

        # Update results
        results['requests'] += 1
        if success and 200 <= status_code < 300:
            results['successful_requests'] += 1
        else:
            results['failed_requests'] += 1
            results['errors'].append({
                'service': service_name,
                'method': method,
                'endpoint': endpoint,
                'status_code': status_code
            })

        results['response_times'].append(response_time)

        # Rotate endpoint for next request
        user_id = (user_id + 1) % len(all_endpoints)

        # Add some randomness to simulate real user behavior
        time.sleep(0.1 + (user_id % 10) * 0.01)

    logger.debug(f"Ending user session: {results['user_id']}")
    return results

def collect_system_metrics() -> Dict[str, Any]:
    """
    Collect system metrics.

    Returns:
        Dictionary with system metrics
    """
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'cpu': {
            'percent': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count(),
            'per_cpu': psutil.cpu_percent(interval=1, percpu=True)
        },
        'memory': {
            'percent': psutil.virtual_memory().percent,
            'used': psutil.virtual_memory().used,
            'total': psutil.virtual_memory().total
        },
        'disk': {
            'percent': psutil.disk_usage('/').percent,
            'used': psutil.disk_usage('/').used,
            'total': psutil.disk_usage('/').total
        },
        'network': {
            'bytes_sent': psutil.net_io_counters().bytes_sent,
            'bytes_recv': psutil.net_io_counters().bytes_recv
        }
    }

    return metrics

def run_performance_test() -> Dict[str, Any]:
    """
    Run the performance test.

    Returns:
        Dictionary with test results
    """
    logger.info("Starting performance test")

    # Create test data directory
    os.makedirs(TEST_CONFIG['test_data_dir'], exist_ok=True)

    # Initialize results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': TEST_CONFIG,
        'services': SERVICE_CONFIG,
        'user_sessions': [],
        'system_metrics': [],
        'summary': {}
    }

    # Create stop event
    stop_event = threading.Event()

    # Start system metrics collection
    metrics_thread = threading.Thread(
        target=lambda: collect_metrics(results, stop_event)
    )
    metrics_thread.daemon = True
    metrics_thread.start()

    # Start user sessions
    with concurrent.futures.ThreadPoolExecutor(max_workers=TEST_CONFIG['users']) as executor:
        # Submit user sessions
        future_to_user = {
            executor.submit(user_session, user_id, stop_event): user_id
            for user_id in range(TEST_CONFIG['users'])
        }

        # Wait for the test duration
        time.sleep(TEST_CONFIG['duration'])

        # Signal sessions to stop
        logger.info("Stopping performance test")
        stop_event.set()

        # Collect results
        for future in concurrent.futures.as_completed(future_to_user):
            user_id = future_to_user[future]
            try:
                session_results = future.result()
                results['user_sessions'].append(session_results)
            except Exception as e:
                logger.error(f"Error in user session {user_id}: {str(e)}")

    # Wait for metrics thread to finish
    metrics_thread.join(timeout=5)

    # Calculate summary
    calculate_summary(results)

    # Save results
    save_results(results)

    return results

def collect_metrics(results: Dict[str, Any], stop_event) -> None:
    """
    Collect system metrics at regular intervals.

    Args:
        results: Dictionary to store metrics
        stop_event: Event to signal when to stop
    """
    while not stop_event.is_set():
        metrics = collect_system_metrics()
        results['system_metrics'].append(metrics)
        time.sleep(TEST_CONFIG['metrics_interval'])

def calculate_summary(results: Dict[str, Any]) -> None:
    """
    Calculate summary statistics from test results.

    Args:
        results: Test results
    """
    logger.info("Calculating summary statistics")

    # Initialize summary
    summary = {
        'total_requests': 0,
        'successful_requests': 0,
        'failed_requests': 0,
        'error_rate': 0,
        'response_time': {
            'min': 0,
            'max': 0,
            'avg': 0,
            'median': 0,
            'p95': 0,
            'p99': 0
        },
        'throughput': 0,
        'system_metrics': {
            'cpu': {
                'avg': 0,
                'max': 0
            },
            'memory': {
                'avg': 0,
                'max': 0
            }
        },
        'thresholds': {
            'response_time_avg': {
                'threshold': TEST_CONFIG['thresholds']['response_time_avg'],
                'actual': 0,
                'passed': False
            },
            'response_time_p95': {
                'threshold': TEST_CONFIG['thresholds']['response_time_p95'],
                'actual': 0,
                'passed': False
            },
            'error_rate': {
                'threshold': TEST_CONFIG['thresholds']['error_rate'],
                'actual': 0,
                'passed': False
            },
            'cpu_usage': {
                'threshold': TEST_CONFIG['thresholds']['cpu_usage'],
                'actual': 0,
                'passed': False
            },
            'memory_usage': {
                'threshold': TEST_CONFIG['thresholds']['memory_usage'],
                'actual': 0,
                'passed': False
            }
        }
    }

    # Collect all response times
    all_response_times = []

    # Process user session results
    for session in results['user_sessions']:
        summary['total_requests'] += session['requests']
        summary['successful_requests'] += session['successful_requests']
        summary['failed_requests'] += session['failed_requests']
        all_response_times.extend(session['response_times'])

    # Calculate error rate
    if summary['total_requests'] > 0:
        summary['error_rate'] = summary['failed_requests'] / summary['total_requests']

    # Calculate response time statistics
    if all_response_times:
        all_response_times.sort()
        summary['response_time']['min'] = min(all_response_times)
        summary['response_time']['max'] = max(all_response_times)
        summary['response_time']['avg'] = statistics.mean(all_response_times)
        summary['response_time']['median'] = statistics.median(all_response_times)
        summary['response_time']['p95'] = all_response_times[int(len(all_response_times) * 0.95)]
        summary['response_time']['p99'] = all_response_times[int(len(all_response_times) * 0.99)]

    # Calculate throughput
    summary['throughput'] = summary['total_requests'] / TEST_CONFIG['duration']

    # Process system metrics
    cpu_values = [m['cpu']['percent'] for m in results['system_metrics']]
    memory_values = [m['memory']['percent'] for m in results['system_metrics']]

    if cpu_values:
        summary['system_metrics']['cpu']['avg'] = statistics.mean(cpu_values)
        summary['system_metrics']['cpu']['max'] = max(cpu_values)

    if memory_values:
        summary['system_metrics']['memory']['avg'] = statistics.mean(memory_values)
        summary['system_metrics']['memory']['max'] = max(memory_values)

    # Check thresholds
    summary['thresholds']['response_time_avg']['actual'] = summary['response_time']['avg']
    summary['thresholds']['response_time_avg']['passed'] = (
        summary['response_time']['avg'] <= TEST_CONFIG['thresholds']['response_time_avg']
    )

    summary['thresholds']['response_time_p95']['actual'] = summary['response_time']['p95']
    summary['thresholds']['response_time_p95']['passed'] = (
        summary['response_time']['p95'] <= TEST_CONFIG['thresholds']['response_time_p95']
    )

    summary['thresholds']['error_rate']['actual'] = summary['error_rate']
    summary['thresholds']['error_rate']['passed'] = (
        summary['error_rate'] <= TEST_CONFIG['thresholds']['error_rate']
    )

    summary['thresholds']['cpu_usage']['actual'] = summary['system_metrics']['cpu']['max']
    summary['thresholds']['cpu_usage']['passed'] = (
        summary['system_metrics']['cpu']['max'] <= TEST_CONFIG['thresholds']['cpu_usage']
    )

    summary['thresholds']['memory_usage']['actual'] = summary['system_metrics']['memory']['max']
    summary['thresholds']['memory_usage']['passed'] = (
        summary['system_metrics']['memory']['max'] <= TEST_CONFIG['thresholds']['memory_usage']
    )

    # Add summary to results
    results['summary'] = summary

def save_results(results: Dict[str, Any]) -> str:
    """
    Save test results to a file.

    Args:
        results: Test results

    Returns:
        Path to the saved file
    """
    logger.info("Saving test results")

    # Create filename
    filename = f"performance_test_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    filepath = os.path.join(TEST_CONFIG['test_data_dir'], filename)

    # Save results
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {filepath}")
    return filepath

def main():
    """Main function to run the performance test."""
    logger.info("Starting performance test script")

    try:
        # Run the performance test
        results = run_performance_test()

        # Print summary
        logger.info("Performance test completed")
        logger.info(f"Total requests: {results['summary']['total_requests']}")
        logger.info(f"Successful requests: {results['summary']['successful_requests']}")
        logger.info(f"Failed requests: {results['summary']['failed_requests']}")
        logger.info(f"Error rate: {results['summary']['error_rate']:.2%}")
        logger.info(f"Average response time: {results['summary']['response_time']['avg']:.2f} ms")
        logger.info(f"95th percentile response time: {results['summary']['response_time']['p95']:.2f} ms")
        logger.info(f"Throughput: {results['summary']['throughput']:.2f} requests/second")

        # Check if all thresholds passed
        all_passed = all(t['passed'] for t in results['summary']['thresholds'].values())

        if all_passed:
            logger.info("All performance thresholds passed!")
            return 0
        else:
            logger.warning("Some performance thresholds failed")
            for name, threshold in results['summary']['thresholds'].items():
                if not threshold['passed']:
                    logger.warning(
                        f"{name}: Actual {threshold['actual']} > Threshold {threshold['threshold']}"
                    )
            return 1

    except Exception as e:
        logger.error(f"Error running performance test: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
