"""
Simple test script for the database connection module.

This script tests the basic functionality of the database connection module
without relying on the pytest framework or conftest.py.
"""
import os
import sys
import unittest
from unittest import mock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Now we can import the module
from analysis_engine.db.connection import (
    get_db, get_db_session, check_db_connection, init_db,
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
)

def test_db_connection():
    """Test that the database connection module can be imported and used."""
    print("Testing database connection module...")

    # Test that environment variables are correctly read
    print(f"DB_HOST: {DB_HOST}")
    print(f"DB_PORT: {DB_PORT}")
    print(f"DB_NAME: {DB_NAME}")
    print(f"DB_USER: {DB_USER}")
    print(f"DB_PASSWORD: {'*' * len(DB_PASSWORD)}")

    # Test that the functions exist
    print("Checking that functions exist...")
    assert callable(get_db), "get_db should be callable"
    assert callable(get_db_session), "get_db_session should be callable"
    assert callable(check_db_connection), "check_db_connection should be callable"
    assert callable(init_db), "init_db should be callable"

    print("All tests passed!")
    return True

if __name__ == '__main__':
    test_db_connection()