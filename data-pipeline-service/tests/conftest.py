"""
Configuration file for pytest to setup proper import paths and environment.
"""
import os
import sys
import pytest

# Add the project root to the path to allow importing common-lib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import common fixtures for reuse across test modules
"""
Test configuration for data-pipeline-service tests.
"""
import logging
from pathlib import Path

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Define test fixtures that can be reused across test modules
@pytest.fixture
def test_data_dir():
    """Fixture providing the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_price_data():
    """Fixture providing a sample of price data for testing."""
    return {
        "EURUSD": [
            {"timestamp": "2025-01-01T00:00:00", "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.105, "volume": 1000},
            {"timestamp": "2025-01-01T01:00:00", "open": 1.105, "high": 1.12, "low": 1.1, "close": 1.115, "volume": 1200},
            {"timestamp": "2025-01-01T02:00:00", "open": 1.115, "high": 1.125, "low": 1.11, "close": 1.12, "volume": 1100},
        ]
    }


@pytest.fixture
def sample_invalid_data():
    """Fixture providing examples of invalid data for testing validation."""
    return {
        "missing_fields": {"timestamp": "2025-01-01T00:00:00", "open": 1.1, "high": 1.11, "low": 1.09},
        "invalid_timestamp": {"timestamp": "not-a-timestamp", "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.105, "volume": 1000},
        "negative_price": {"timestamp": "2025-01-01T00:00:00", "open": -1.1, "high": 1.11, "low": 1.09, "close": 1.105, "volume": 1000},
        "high_lower_than_low": {"timestamp": "2025-01-01T00:00:00", "open": 1.1, "high": 1.01, "low": 1.09, "close": 1.105, "volume": 1000},
    }
