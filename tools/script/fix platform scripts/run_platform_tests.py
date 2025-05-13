#!/usr/bin/env python3
"""
Forex Trading Platform Test Runner

This script runs tests for the reorganized forex trading platform structure.
It executes unit tests, integration tests, and health checks for all services.

Usage:
python run_platform_tests.py [--project-root <project_root>] [--service <service_name>] [--test-type <test_type>]
"""

import os
import sys
import argparse
import logging
import subprocess
import concurrent.futures
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROJECT_ROOT = r"D:\MD\forex_trading_platform"

class PlatformTestRunner:
    """Runs tests for the forex trading platform."""

    def __init__(self, project_root: str, service_name: Optional[str] = None, test_type: Optional[str] = None):
        """
        Initialize the test runner.

        Args:
            project_root: Root directory of the project
            service_name: Name of the service to test (None for all services)
            test_type: Type of tests to run (unit, integration, health, all)
        """
        self.project_root = project_root
        self.service_name = service_name
        self.test_type = test_type or 'all'
        self.services = []
        self.results = {}

    def identify_services(self) -> None:
        """Identify services in the project based on directory structure."""
        logger.info("Identifying services...")

        # Look for service directories
        for item in os.listdir(self.project_root):
            item_path = os.path.join(self.project_root, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if it's likely a service
                if (
                    item.endswith('-service') or
                    item.endswith('_service') or
                    item.endswith('-api') or
                    item.endswith('-engine') or
                    'service' in item.lower() or
                    'api' in item.lower()
                ):
                    self.services.append(item)

        logger.info(f"Identified {len(self.services)} services")

    def run_unit_tests(self, service_name: str) -> Dict[str, Any]:
        """
        Run unit tests for a service.

        Args:
            service_name: Name of the service

        Returns:
            Test results
        """
        logger.info(f"Running unit tests for {service_name}...")

        service_path = os.path.join(self.project_root, service_name)
        tests_path = os.path.join(service_path, 'tests')

        # Check if tests directory exists
        if not os.path.exists(tests_path):
            logger.warning(f"No tests directory found for {service_name}")
            return {
                'service': service_name,
                'test_type': 'unit',
                'status': 'skipped',
                'message': 'No tests directory found'
            }

        # Check if pytest is installed
        try:
            import pytest
            pytest_installed = True
        except ImportError:
            pytest_installed = False
            logger.warning("pytest is not installed, skipping tests")
            return {
                'service': service_name,
                'test_type': 'unit',
                'status': 'skipped',
                'message': 'pytest is not installed'
            }

        # Run pytest if installed
        if pytest_installed:
            try:
                # Print command for debugging
                logger.info(f"Running command: pytest {tests_path} -v")

                # Run pytest directly
                result = subprocess.run(
                    ['pytest', tests_path, '-v'],
                    capture_output=True,
                    text=True,
                    check=False
                )

                # Log output for debugging
                logger.info(f"Pytest stdout: {result.stdout}")
                if result.stderr:
                    logger.error(f"Pytest stderr: {result.stderr}")

                # Parse test results
                if result.returncode == 0:
                    return {
                        'service': service_name,
                        'test_type': 'unit',
                        'status': 'passed',
                        'output': result.stdout
                    }
                else:
                    return {
                        'service': service_name,
                        'test_type': 'unit',
                        'status': 'failed',
                        'output': result.stdout,
                        'error': result.stderr
                    }
            except Exception as e:
                logger.error(f"Error running unit tests for {service_name}: {e}")
                return {
                    'service': service_name,
                    'test_type': 'unit',
                    'status': 'error',
                    'message': str(e)
                }

    def run_integration_tests(self, service_name: str) -> Dict[str, Any]:
        """
        Run integration tests for a service.

        Args:
            service_name: Name of the service

        Returns:
            Test results
        """
        logger.info(f"Running integration tests for {service_name}...")

        service_path = os.path.join(self.project_root, service_name)
        integration_tests_path = os.path.join(service_path, 'tests', 'integration')

        # Check if integration tests directory exists
        if not os.path.exists(integration_tests_path):
            logger.warning(f"No integration tests directory found for {service_name}")
            return {
                'service': service_name,
                'test_type': 'integration',
                'status': 'skipped',
                'message': 'No integration tests directory found'
            }

        # Run pytest
        try:
            result = subprocess.run(
                ['pytest', integration_tests_path, '-v'],
                capture_output=True,
                text=True,
                check=False
            )

            # Parse test results
            if result.returncode == 0:
                return {
                    'service': service_name,
                    'test_type': 'integration',
                    'status': 'passed',
                    'output': result.stdout
                }
            else:
                return {
                    'service': service_name,
                    'test_type': 'integration',
                    'status': 'failed',
                    'output': result.stdout,
                    'error': result.stderr
                }
        except Exception as e:
            logger.error(f"Error running integration tests for {service_name}: {e}")
            return {
                'service': service_name,
                'test_type': 'integration',
                'status': 'error',
                'message': str(e)
            }

    def run_health_check(self, service_name: str) -> Dict[str, Any]:
        """
        Run health check for a service.

        Args:
            service_name: Name of the service

        Returns:
            Health check results
        """
        logger.info(f"Running health check for {service_name}...")

        service_path = os.path.join(self.project_root, service_name)

        # Look for health check script
        health_check_paths = [
            os.path.join(service_path, 'health_check.py'),
            os.path.join(service_path, 'core', 'health_check.py'),
            os.path.join(service_path, 'api', 'health.py'),
            os.path.join(service_path, 'api', 'health_check.py')
        ]

        health_check_path = None
        for path in health_check_paths:
            if os.path.exists(path):
                health_check_path = path
                break

        if not health_check_path:
            logger.warning(f"No health check script found for {service_name}")
            return {
                'service': service_name,
                'test_type': 'health',
                'status': 'skipped',
                'message': 'No health check script found'
            }

        # Run health check
        try:
            result = subprocess.run(
                ['python', health_check_path],
                capture_output=True,
                text=True,
                check=False
            )

            # Parse health check results
            if result.returncode == 0:
                return {
                    'service': service_name,
                    'test_type': 'health',
                    'status': 'passed',
                    'output': result.stdout
                }
            else:
                return {
                    'service': service_name,
                    'test_type': 'health',
                    'status': 'failed',
                    'output': result.stdout,
                    'error': result.stderr
                }
        except Exception as e:
            logger.error(f"Error running health check for {service_name}: {e}")
            return {
                'service': service_name,
                'test_type': 'health',
                'status': 'error',
                'message': str(e)
            }

    def run_tests(self) -> Dict[str, Any]:
        """
        Run tests for the forex trading platform.

        Returns:
            Test results
        """
        logger.info("Starting platform tests...")

        # Identify services
        self.identify_services()

        if not self.services:
            logger.info("No services found")
            return {}

        # Filter services if a specific service was specified
        if self.service_name:
            if self.service_name in self.services:
                services_to_test = [self.service_name]
            else:
                logger.error(f"Service not found: {self.service_name}")
                return {}
        else:
            services_to_test = self.services

        # Run tests for each service
        for service in services_to_test:
            self.results[service] = {}

            # Run unit tests
            if self.test_type in ['unit', 'all']:
                self.results[service]['unit'] = self.run_unit_tests(service)

            # Run integration tests
            if self.test_type in ['integration', 'all']:
                self.results[service]['integration'] = self.run_integration_tests(service)

            # Run health check
            if self.test_type in ['health', 'all']:
                self.results[service]['health'] = self.run_health_check(service)

        logger.info("Platform tests complete")
        return self.results

    def generate_report(self) -> str:
        """
        Generate a test report.

        Returns:
            Test report
        """
        logger.info("Generating test report...")

        # Count test results
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        error_tests = 0

        for service, tests in self.results.items():
            for test_type, result in tests.items():
                total_tests += 1
                if result['status'] == 'passed':
                    passed_tests += 1
                elif result['status'] == 'failed':
                    failed_tests += 1
                elif result['status'] == 'skipped':
                    skipped_tests += 1
                elif result['status'] == 'error':
                    error_tests += 1

        # Generate report
        report = f"""
Forex Trading Platform Test Report
=================================

Summary
-------
Total tests: {total_tests}
Passed: {passed_tests}
Failed: {failed_tests}
Skipped: {skipped_tests}
Errors: {error_tests}

Details
-------
"""

        for service, tests in self.results.items():
            report += f"\n{service}\n{'-' * len(service)}\n"

            for test_type, result in tests.items():
                report += f"  {test_type}: {result['status']}\n"

                if result['status'] == 'failed' or result['status'] == 'error':
                    if 'message' in result:
                        report += f"    Message: {result['message']}\n"
                    if 'error' in result:
                        report += f"    Error: {result['error']}\n"
                    if 'output' in result:
                        report += f"    Output: {result['output']}\n"

        return report

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run platform tests")
    parser.add_argument(
        "--project-root",
        default=DEFAULT_PROJECT_ROOT,
        help="Root directory of the project"
    )
    parser.add_argument(
        "--service",
        help="Name of the service to test (default: all services)"
    )
    parser.add_argument(
        "--test-type",
        choices=['unit', 'integration', 'health', 'all'],
        default='all',
        help="Type of tests to run (default: all)"
    )
    args = parser.parse_args()

    # Run tests
    test_runner = PlatformTestRunner(args.project_root, args.service, args.test_type)
    test_runner.run_tests()

    # Generate report
    report = test_runner.generate_report()
    print(report)

    # Save report to file
    output_dir = os.path.join(args.project_root, 'tools', 'output')
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'test_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"Test report saved to {report_path}")

if __name__ == "__main__":
    main()
