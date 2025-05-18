"""
Enable mocking for database benchmarks.

This script enables mocking for database benchmarks by setting the USE_MOCKS flag to True.
"""
import sys
import os

# Add common-lib to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common-lib'))

# Import the database module
import common_lib.database

# Enable mocking
common_lib.database.USE_MOCKS = True

print("Database mocking enabled for benchmarks.")