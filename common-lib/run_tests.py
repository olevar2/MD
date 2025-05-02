#!/usr/bin/env python
"""
Test runner for common-lib.
"""

import asyncio
import unittest
import os
import sys
from pathlib import Path

# Add parent directory to path to ensure imports work correctly
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


def discover_and_run_tests():
    """Discover and run all tests."""
    # Discover tests in the tests directory
    test_dir = os.path.join(os.path.dirname(__file__), "tests")
    test_suite = unittest.defaultTestLoader.discover(test_dir)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(test_suite)


if __name__ == "__main__":
    result = discover_and_run_tests()
    sys.exit(not result.wasSuccessful())
