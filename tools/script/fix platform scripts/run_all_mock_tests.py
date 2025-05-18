"""
Script to run all the mock implementation tests.

This script runs all the tests for the mock implementation of the database utilities.
"""
import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_test(test_file):
    """Run a test file."""
    logger.info(f"Running test: {test_file}")
    
    # Run the test
    result = subprocess.run(
        ["python", test_file],
        capture_output=True,
        text=True,
    )
    
    # Check the result
    if result.returncode == 0:
        logger.info(f"Test {test_file} passed!")
    else:
        logger.error(f"Test {test_file} failed!")
        logger.error(f"Output: {result.stdout}")
        logger.error(f"Error: {result.stderr}")
    
    return result.returncode == 0


def main():
    """Main function."""
    logger.info("Running all mock implementation tests...")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of test files
    test_files = [
        os.path.join(current_dir, "test_mock_implementation_fixed.py"),
        os.path.join(current_dir, "test_improved_mock_implementation.py"),
    ]
    
    # Run all tests
    all_passed = True
    for test_file in test_files:
        if not run_test(test_file):
            all_passed = False
    
    # Print the result
    if all_passed:
        logger.info("All tests passed!")
    else:
        logger.error("Some tests failed!")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())