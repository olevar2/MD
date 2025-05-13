"""
Test fixtures for risk management service tests.
These fixtures provide test data that can be used across multiple test modules.
"""
import json
import os
import pytest

# Path to the fixtures directory from this file
FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def market_data_fixture():
    """Fixture that provides sample market data for testing."""
    fixture_path = os.path.join(FIXTURES_DIR, 'market_data.json')
    
    # If the file doesn't exist, create a default fixture
    if not os.path.exists(fixture_path):
        data = {
            "EUR/USD": {
                "bid": 1.1050,
                "ask": 1.1052,
                "daily_volatility": 0.0045,
                "weekly_range": [1.0950, 1.1150]
            },
            "GBP/USD": {
                "bid": 1.3020,
                "ask": 1.3023,
                "daily_volatility": 0.0065,
                "weekly_range": [1.2900, 1.3200]
            },
            "USD/JPY": {
                "bid": 108.20,
                "ask": 108.22,
                "daily_volatility": 0.0055,
                "weekly_range": [107.50, 109.00]
            }
        }
        
        os.makedirs(os.path.dirname(fixture_path), exist_ok=True)
        with open(fixture_path, 'w') as f:
            json.dump(data, f, indent=4)
    
    # Read and return the fixture data
    with open(fixture_path, 'r') as f:
        return json.load(f)

@pytest.fixture
def correlation_matrix_fixture():
    """Fixture that provides a sample correlation matrix for testing."""
    return {
        "EUR/USD": {
            "EUR/USD": 1.0,
            "GBP/USD": 0.85,
            "USD/JPY": -0.65
        },
        "GBP/USD": {
            "EUR/USD": 0.85,
            "GBP/USD": 1.0,
            "USD/JPY": -0.55
        },
        "USD/JPY": {
            "EUR/USD": -0.65,
            "GBP/USD": -0.55,
            "USD/JPY": 1.0
        }
    }
