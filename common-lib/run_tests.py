#!/usr/bin/env python
"""
Test runner for common-lib.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import subprocess

# Add parent directory to path to ensure imports work correctly
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


def run_unit_tests():
    """Run unit tests using pytest."""
    print("Running unit tests...")
    start_time = time.time()

    # Run pytest on the tests directory, excluding integration tests
    result = subprocess.run(
        ["pytest", "tests", "-k", "not integration"],
        cwd=os.path.dirname(__file__)
    )

    elapsed_time = time.time() - start_time

    if result.returncode == 0:
        print(f"Unit tests passed in {elapsed_time:.2f} seconds.")
    else:
        print(f"Unit tests failed in {elapsed_time:.2f} seconds.")

    return result.returncode == 0


def run_integration_tests():
    """Run integration tests using pytest."""
    print("Running integration tests...")
    start_time = time.time()

    # Run pytest on the integration tests directory
    result = subprocess.run(
        ["pytest", "tests/integration"],
        cwd=os.path.dirname(__file__)
    )

    elapsed_time = time.time() - start_time

    if result.returncode == 0:
        print(f"Integration tests passed in {elapsed_time:.2f} seconds.")
    else:
        print(f"Integration tests failed in {elapsed_time:.2f} seconds.")

    return result.returncode == 0


def run_all_tests():
    """Run all tests using pytest."""
    print("Running all tests...")
    start_time = time.time()

    # Run pytest on the tests directory
    result = subprocess.run(
        ["pytest", "tests"],
        cwd=os.path.dirname(__file__)
    )

    elapsed_time = time.time() - start_time

    if result.returncode == 0:
        print(f"All tests passed in {elapsed_time:.2f} seconds.")
    else:
        print(f"Tests failed in {elapsed_time:.2f} seconds.")

    return result.returncode == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for common-lib")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--unit", action="store_true", help="Run unit tests only")
    group.add_argument("--integration", action="store_true", help="Run integration tests only")
    group.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    if args.unit:
        success = run_unit_tests()
    elif args.integration:
        success = run_integration_tests()
    elif args.all:
        success = run_all_tests()
    else:
        # Default to running unit tests
        success = run_unit_tests()

    sys.exit(0 if success else 1)
