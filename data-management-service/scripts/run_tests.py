#!/usr/bin/env python
"""
Run tests for the Historical Data Management service.

This script runs the tests for the service.
"""

import argparse
import logging
import os
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tests")
    
    parser.add_argument(
        "--test-path",
        type=str,
        default="tests",
        help="Path to test directory (default: tests)"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report (default: False)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Build command
    cmd = [sys.executable, "-m", "pytest", args.test_path, "-v"]
    
    if args.coverage:
        cmd.extend(["--cov=data_management_service", "--cov-report=term", "--cov-report=html"])
    
    # Run tests
    logger.info(f"Running tests: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Print output
    for line in process.stdout:
        print(line, end="")
    
    # Wait for process to complete
    process.wait()
    
    # Print errors
    if process.returncode != 0:
        logger.error(f"Tests failed with exit code {process.returncode}")
        for line in process.stderr:
            print(line, end="")
    else:
        logger.info("Tests passed")
    
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
