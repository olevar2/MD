"""
Test db connection module.

This module provides functionality for...
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the database connection module
sys.path.append('D:/MD/forex_trading_platform/analysis-engine-service')
from analysis_engine.db import initialize_database, check_db_connection

# Initialize the database
initialize_database()

# Check the database connection
connection_ok = check_db_connection()

print(f"Database connection test: {connection_ok}")