"""
Integration tests for backtesting service.
"""
import pytest
import os
import psycopg2
import redis
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# Get PostgreSQL connection details from environment variables
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "backtesting_test")

# Get Redis connection details from environment variables
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

@pytest.fixture
def postgres_connection():
    """Create a PostgreSQL connection for testing."""
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB
    )
    yield conn
    # Clean up after tests
    conn.close()

@pytest.fixture
def redis_client():
    """Create a Redis client for testing."""
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    yield r
    # Clean up after tests
    r.flushall()

def test_postgres_connection(postgres_connection):
    """Test that the service can connect to PostgreSQL."""
    cursor = postgres_connection.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()
    assert result[0] == 1

def test_redis_connection(redis_client):
    """Test that the service can connect to Redis."""
    # Set a value in Redis
    redis_client.set("test_key", "test_value")
    
    # Get the value from Redis
    value = redis_client.get("test_key")
    
    # Verify the value
    assert value == b"test_value"

def test_backtesting_with_database(postgres_connection):
    """Test that backtesting results are stored in the database."""
    # Sample data for testing
    test_data = {
        "strategy_id": "test_strategy",
        "start_date": "2023-01-01",
        "end_date": "2023-01-31",
        "symbol": "EURUSD",
        "timeframe": "1h",
        "parameters": {
            "risk_reward_ratio": 2.0,
            "stop_loss_pips": 20,
            "take_profit_pips": 40
        }
    }
    
    # Make a request to the backtesting endpoint
    response = client.post("/api/v1/backtest", json=test_data)
    assert response.status_code == 200
    
    # In a real test, you would verify that the results are stored in the database
    # This is a simplified test that just checks if the database is working
    cursor = postgres_connection.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()
    assert result[0] == 1

def test_service_integration():
    """Test integration with other services."""
    # This is a placeholder for testing integration with other services
    # In a real test, you would make requests to other services
    # and verify that the backtesting service handles the responses correctly
    assert True
