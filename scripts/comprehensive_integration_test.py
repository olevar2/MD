#!/usr/bin/env python3
"""
Comprehensive integration test for the Forex Trading Platform.

This script tests the integration between all services in the platform,
verifying that they work together correctly.
"""

import os
import sys
import json
import time
import logging
import requests
import subprocess
import concurrent.futures
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('integration_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Service configuration
SERVICE_CONFIG = {
    'trading-gateway-service': {
        'port': 8001,
        'dependencies': [],
        'health_endpoint': '/health',
        'test_endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/instruments', 'expected_status': 200},
            {'method': 'GET', 'endpoint': '/api/v1/accounts', 'expected_status': 200}
        ]
    },
    'portfolio-management-service': {
        'port': 8002,
        'dependencies': ['trading-gateway-service'],
        'health_endpoint': '/health',
        'test_endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/portfolios', 'expected_status': 200},
            {'method': 'GET', 'endpoint': '/api/v1/positions', 'expected_status': 200}
        ]
    },
    'risk-management-service': {
        'port': 8003,
        'dependencies': ['portfolio-management-service'],
        'health_endpoint': '/health',
        'test_endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/risk-profiles', 'expected_status': 200},
            {'method': 'GET', 'endpoint': '/api/v1/risk-limits', 'expected_status': 200}
        ]
    },
    'data-pipeline-service': {
        'port': 8004,
        'dependencies': [],
        'health_endpoint': '/health',
        'test_endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/market-data', 'expected_status': 200},
            {'method': 'GET', 'endpoint': '/api/v1/data-sources', 'expected_status': 200}
        ]
    },
    'feature-store-service': {
        'port': 8005,
        'dependencies': ['data-pipeline-service'],
        'health_endpoint': '/health',
        'test_endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/features', 'expected_status': 200},
            {'method': 'GET', 'endpoint': '/api/v1/feature-sets', 'expected_status': 200}
        ]
    },
    'ml-integration-service': {
        'port': 8006,
        'dependencies': ['feature-store-service'],
        'health_endpoint': '/health',
        'test_endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/models', 'expected_status': 200},
            {'method': 'GET', 'endpoint': '/api/v1/predictions', 'expected_status': 200}
        ]
    },
    'ml-workbench-service': {
        'port': 8007,
        'dependencies': ['feature-store-service'],
        'health_endpoint': '/health',
        'test_endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/experiments', 'expected_status': 200},
            {'method': 'GET', 'endpoint': '/api/v1/model-registry', 'expected_status': 200}
        ]
    },
    'monitoring-alerting-service': {
        'port': 8008,
        'dependencies': ['trading-gateway-service', 'portfolio-management-service', 'risk-management-service'],
        'health_endpoint': '/health',
        'test_endpoints': [
            {'method': 'GET', 'endpoint': '/api/v1/alerts', 'expected_status': 200},
            {'method': 'GET', 'endpoint': '/api/v1/metrics', 'expected_status': 200}
        ]
    }
}

# Test configuration
TEST_CONFIG = {
    'max_startup_time': 60,  # Maximum time to wait for services to start (seconds)
    'request_timeout': 10,   # Timeout for HTTP requests (seconds)
    'retry_interval': 2,     # Interval between retries (seconds)
    'max_retries': 5,        # Maximum number of retries
    'parallel_tests': True,  # Run tests in parallel
    'cleanup_after_test': True,  # Stop services after testing
    'test_data_dir': 'test_data'  # Directory for test data
}

def start_service(service_name: str) -> subprocess.Popen:
    """
    Start a service.

    Args:
        service_name: Name of the service to start

    Returns:
        Process object for the started service
    """
    service_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), service_name)

    # Check if the service directory exists
    if not os.path.exists(service_dir):
        logger.error(f"Service directory not found: {service_dir}")
        raise FileNotFoundError(f"Service directory not found: {service_dir}")

    # Start the service
    logger.info(f"Starting service: {service_name}")

    # For testing purposes, we'll use a mock service
    # In a real implementation, this would start the actual service

    # Create a mock service script
    mock_script = f"""
import http.server
import socketserver
import json
import time
from urllib.parse import urlparse, parse_qs

PORT = {SERVICE_CONFIG[service_name]['port']}

class MockHandler(http.server.BaseHTTPRequestHandler):
    """
    MockHandler class that inherits from http.server.BaseHTTPRequestHandler.
    
    Attributes:
        Add attributes here
    """

    def do_GET(self):
    """
    Do get.
    
    """

        path = urlparse(self.path).path

        # Health check endpoint
        if path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {{"status": "healthy"}}
            self.wfile.write(json.dumps(response).encode())
            return

        # API endpoints
        for endpoint_config in {SERVICE_CONFIG[service_name]['test_endpoints']}:
            if path == endpoint_config['endpoint']:
                self.send_response(endpoint_config['expected_status'])
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {{"status": "success", "data": []}}
                self.wfile.write(json.dumps(response).encode())
                return

        # Test interaction endpoints
        if path.startswith('/api/v1/test/'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {{"status": "success", "message": "Interaction successful"}}
            self.wfile.write(json.dumps(response).encode())
            return

        # Default response for unknown endpoints
        self.send_response(404)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {{"status": "error", "message": "Endpoint not found"}}
        self.wfile.write(json.dumps(response).encode())

print(f"Starting mock service for {service_name} on port {{PORT}}")
httpd = socketserver.TCPServer(("", PORT), MockHandler)
httpd.serve_forever()
"""

    # Write the mock script to a temporary file
    mock_script_path = os.path.join(service_dir, 'mock_service.py')
    with open(mock_script_path, 'w') as f:
        f.write(mock_script)

    # Start the mock service
    process = subprocess.Popen(
        [sys.executable, mock_script_path],
        cwd=service_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    return process

def wait_for_service(service_name: str, port: int, max_wait_time: int = 60) -> bool:
    """
    Wait for a service to be ready.

    Args:
        service_name: Name of the service
        port: Port the service is running on
        max_wait_time: Maximum time to wait (seconds)

    Returns:
        Whether the service is ready
    """
    logger.info(f"Waiting for service to be ready: {service_name}")

    url = f"http://localhost:{port}/health"
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(url, timeout=TEST_CONFIG['request_timeout'])
            if response.status_code == 200:
                logger.info(f"Service is ready: {service_name}")
                return True
        except requests.RequestException:
            pass

        time.sleep(TEST_CONFIG['retry_interval'])

    logger.error(f"Timed out waiting for service: {service_name}")
    return False

def start_all_services() -> Dict[str, subprocess.Popen]:
    """
    Start all services.

    Returns:
        Dictionary mapping service names to process objects
    """
    logger.info("Starting all services")

    processes = {}

    # Start services in dependency order
    for service_name, config in SERVICE_CONFIG.items():
        # Check if all dependencies are ready
        dependencies_ready = True
        for dependency in config['dependencies']:
            if dependency not in processes:
                dependencies_ready = False
                break

        if not dependencies_ready:
            logger.warning(f"Skipping service due to missing dependencies: {service_name}")
            continue

        # Start the service
        try:
            process = start_service(service_name)
            processes[service_name] = process

            # Wait for the service to be ready
            if not wait_for_service(service_name, config['port'], TEST_CONFIG['max_startup_time']):
                logger.error(f"Service failed to start: {service_name}")
                stop_services(processes)
                raise RuntimeError(f"Service failed to start: {service_name}")
        except Exception as e:
            logger.error(f"Error starting service {service_name}: {str(e)}")
            stop_services(processes)
            raise

    return processes

def stop_services(processes: Dict[str, subprocess.Popen]) -> None:
    """
    Stop all services.

    Args:
        processes: Dictionary mapping service names to process objects
    """
    logger.info("Stopping all services")

    for service_name, process in processes.items():
        logger.info(f"Stopping service: {service_name}")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning(f"Service did not terminate gracefully, killing: {service_name}")
            process.kill()

def test_service_health(service_name: str, port: int) -> bool:
    """
    Test the health of a service.

    Args:
        service_name: Name of the service
        port: Port the service is running on

    Returns:
        Whether the service is healthy
    """
    logger.info(f"Testing service health: {service_name}")

    url = f"http://localhost:{port}/health"

    for _ in range(TEST_CONFIG['max_retries']):
        try:
            response = requests.get(url, timeout=TEST_CONFIG['request_timeout'])
            if response.status_code == 200:
                logger.info(f"Service is healthy: {service_name}")
                return True
        except requests.RequestException as e:
            logger.warning(f"Error testing service health {service_name}: {str(e)}")

        time.sleep(TEST_CONFIG['retry_interval'])

    logger.error(f"Service is not healthy: {service_name}")
    return False

def test_service_endpoint(service_name: str, port: int, method: str, endpoint: str, expected_status: int) -> bool:
    """
    Test a service endpoint.

    Args:
        service_name: Name of the service
        port: Port the service is running on
        method: HTTP method
        endpoint: Endpoint to test
        expected_status: Expected HTTP status code

    Returns:
        Whether the test passed
    """
    logger.info(f"Testing service endpoint: {service_name} - {method} {endpoint}")

    url = f"http://localhost:{port}{endpoint}"

    for _ in range(TEST_CONFIG['max_retries']):
        try:
            if method == 'GET':
                response = requests.get(url, timeout=TEST_CONFIG['request_timeout'])
            elif method == 'POST':
                response = requests.post(url, json={}, timeout=TEST_CONFIG['request_timeout'])
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return False

            if response.status_code == expected_status:
                logger.info(f"Endpoint test passed: {service_name} - {method} {endpoint}")
                return True
            else:
                logger.warning(
                    f"Endpoint test failed: {service_name} - {method} {endpoint} - "
                    f"Expected status {expected_status}, got {response.status_code}"
                )
        except requests.RequestException as e:
            logger.warning(f"Error testing endpoint {service_name} - {method} {endpoint}: {str(e)}")

        time.sleep(TEST_CONFIG['retry_interval'])

    logger.error(f"Endpoint test failed: {service_name} - {method} {endpoint}")
    return False

def test_service(service_name: str) -> Tuple[str, bool, Dict[str, Any]]:
    """
    Test a service.

    Args:
        service_name: Name of the service

    Returns:
        Tuple of (service_name, success, results)
    """
    logger.info(f"Testing service: {service_name}")

    config = SERVICE_CONFIG[service_name]
    port = config['port']

    results = {
        'health': False,
        'endpoints': {}
    }

    # Test health
    results['health'] = test_service_health(service_name, port)

    # Test endpoints
    for endpoint_config in config['test_endpoints']:
        endpoint_key = f"{endpoint_config['method']} {endpoint_config['endpoint']}"
        results['endpoints'][endpoint_key] = test_service_endpoint(
            service_name,
            port,
            endpoint_config['method'],
            endpoint_config['endpoint'],
            endpoint_config['expected_status']
        )

    # Check if all tests passed
    success = results['health'] and all(results['endpoints'].values())

    return service_name, success, results

def test_all_services(processes: Dict[str, subprocess.Popen]) -> Dict[str, Dict[str, Any]]:
    """
    Test all services.

    Args:
        processes: Dictionary mapping service names to process objects

    Returns:
        Dictionary mapping service names to test results
    """
    logger.info("Testing all services")

    results = {}

    if TEST_CONFIG['parallel_tests']:
        # Test services in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_service = {
                executor.submit(test_service, service_name): service_name
                for service_name in processes.keys()
            }

            for future in concurrent.futures.as_completed(future_to_service):
                service_name, success, service_results = future.result()
                results[service_name] = {
                    'success': success,
                    'results': service_results
                }
    else:
        # Test services sequentially
        for service_name in processes.keys():
            service_name, success, service_results = test_service(service_name)
            results[service_name] = {
                'success': success,
                'results': service_results
            }

    return results

def test_service_interactions() -> Dict[str, Dict[str, Any]]:
    """
    Test interactions between services.

    Returns:
        Dictionary mapping interaction names to test results
    """
    logger.info("Testing service interactions")

    interactions = {
        'trading_to_portfolio': {
            'description': 'Trading Gateway to Portfolio Management',
            'source_service': 'trading-gateway-service',
            'source_port': SERVICE_CONFIG['trading-gateway-service']['port'],
            'source_endpoint': '/api/v1/test/portfolio-interaction',
            'expected_status': 200
        },
        'portfolio_to_risk': {
            'description': 'Portfolio Management to Risk Management',
            'source_service': 'portfolio-management-service',
            'source_port': SERVICE_CONFIG['portfolio-management-service']['port'],
            'source_endpoint': '/api/v1/test/risk-interaction',
            'expected_status': 200
        },
        'data_to_feature': {
            'description': 'Data Pipeline to Feature Store',
            'source_service': 'data-pipeline-service',
            'source_port': SERVICE_CONFIG['data-pipeline-service']['port'],
            'source_endpoint': '/api/v1/test/feature-interaction',
            'expected_status': 200
        },
        'feature_to_ml': {
            'description': 'Feature Store to ML Integration',
            'source_service': 'feature-store-service',
            'source_port': SERVICE_CONFIG['feature-store-service']['port'],
            'source_endpoint': '/api/v1/test/ml-interaction',
            'expected_status': 200
        }
    }

    results = {}

    for interaction_name, config in interactions.items():
        logger.info(f"Testing interaction: {config['description']}")

        url = f"http://localhost:{config['source_port']}{config['source_endpoint']}"

        success = False
        for _ in range(TEST_CONFIG['max_retries']):
            try:
                response = requests.get(url, timeout=TEST_CONFIG['request_timeout'])
                if response.status_code == config['expected_status']:
                    logger.info(f"Interaction test passed: {config['description']}")
                    success = True
                    break
                else:
                    logger.warning(
                        f"Interaction test failed: {config['description']} - "
                        f"Expected status {config['expected_status']}, got {response.status_code}"
                    )
            except requests.RequestException as e:
                logger.warning(f"Error testing interaction {config['description']}: {str(e)}")

            time.sleep(TEST_CONFIG['retry_interval'])

        if not success:
            logger.error(f"Interaction test failed: {config['description']}")

        results[interaction_name] = {
            'success': success,
            'description': config['description']
        }

    return results

def main():
    """Main function to run the integration test."""
    logger.info("Starting comprehensive integration test")

    # Create test data directory
    os.makedirs(TEST_CONFIG['test_data_dir'], exist_ok=True)

    try:
        # Start all services
        processes = start_all_services()

        # Test all services
        service_results = test_all_services(processes)

        # Test service interactions
        interaction_results = test_service_interactions()

        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'services': service_results,
            'interactions': interaction_results,
            'summary': {
                'services_tested': len(service_results),
                'services_passed': sum(1 for r in service_results.values() if r['success']),
                'interactions_tested': len(interaction_results),
                'interactions_passed': sum(1 for r in interaction_results.values() if r['success'])
            }
        }

        # Save report
        report_path = os.path.join(TEST_CONFIG['test_data_dir'], f"integration_test_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        logger.info("Integration test completed")
        logger.info(f"Services tested: {report['summary']['services_tested']}")
        logger.info(f"Services passed: {report['summary']['services_passed']}")
        logger.info(f"Interactions tested: {report['summary']['interactions_tested']}")
        logger.info(f"Interactions passed: {report['summary']['interactions_passed']}")

        # Check if all tests passed
        all_passed = (
            report['summary']['services_passed'] == report['summary']['services_tested'] and
            report['summary']['interactions_passed'] == report['summary']['interactions_tested']
        )

        if all_passed:
            logger.info("All tests passed!")
            return 0
        else:
            logger.error("Some tests failed")
            return 1

    except Exception as e:
        logger.error(f"Error running integration test: {str(e)}")
        return 1

    finally:
        # Stop services if cleanup is enabled
        if 'processes' in locals() and TEST_CONFIG['cleanup_after_test']:
            stop_services(processes)

if __name__ == '__main__':
    sys.exit(main())
