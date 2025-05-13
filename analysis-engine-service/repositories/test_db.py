"""
Test script for database connection.
"""

from analysis_engine.db.connection import initialize_database, check_db_connection

# Initialize the database
initialize_database()

# Check the database connection
connection_ok = check_db_connection()

print(f"Database connection test: {connection_ok}")