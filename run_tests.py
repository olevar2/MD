"""
Script to run all tests for the forex trading platform.
"""

import os
import sys
import subprocess
import argparse
import time


def run_unit_tests():
    """Run unit tests."""
    print("Running unit tests...")
    start_time = time.time()
    result = subprocess.run(["pytest", "-xvs", "common-lib/tests/templates/service_template"], check=False)
    duration = time.time() - start_time
    
    if result.returncode == 0:
        print(f"Unit tests passed in {duration:.2f} seconds.")
    else:
        print(f"Unit tests failed in {duration:.2f} seconds.")
    
    return result.returncode


def run_integration_tests():
    """Run integration tests."""
    print("Running integration tests...")
    start_time = time.time()
    result = subprocess.run(["pytest", "-xvs", "tests/integration/service_interactions"], check=False)
    duration = time.time() - start_time
    
    if result.returncode == 0:
        print(f"Integration tests passed in {duration:.2f} seconds.")
    else:
        print(f"Integration tests failed in {duration:.2f} seconds.")
    
    return result.returncode


def run_performance_tests():
    """Run performance tests."""
    print("Running performance tests...")
    start_time = time.time()
    result = subprocess.run(["pytest", "-xvs", "tests/performance"], check=False)
    duration = time.time() - start_time
    
    if result.returncode == 0:
        print(f"Performance tests passed in {duration:.2f} seconds.")
    else:
        print(f"Performance tests failed in {duration:.2f} seconds.")
    
    return result.returncode


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run tests for the forex trading platform.")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # If no arguments are provided, run all tests
    if not (args.unit or args.integration or args.performance or args.all):
        args.all = True
    
    # Run tests
    exit_code = 0
    
    if args.unit or args.all:
        unit_exit_code = run_unit_tests()
        exit_code = exit_code or unit_exit_code
    
    if args.integration or args.all:
        integration_exit_code = run_integration_tests()
        exit_code = exit_code or integration_exit_code
    
    if args.performance or args.all:
        performance_exit_code = run_performance_tests()
        exit_code = exit_code or performance_exit_code
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
