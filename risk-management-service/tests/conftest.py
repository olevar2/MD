"""
Configuration for pytest.
This file contains fixtures that will be available to all tests.
"""
import os
import sys
import pytest
import asyncio
from typing import Generator, Any

# Add the parent directory to sys.path to allow importing from the service
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set explicit event loop policy for Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

@pytest.fixture(scope="function")
def event_loop() -> Generator[asyncio.AbstractEventLoop, Any, None]:
    """Create an instance of the default event loop for each test function."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Define fixtures that can be used across all tests
@pytest.fixture
def sample_risk_config():
    """Fixture that provides a sample risk configuration for testing."""
    return {
        "max_drawdown_percent": 5.0,
        "max_leverage": 10.0,
        "max_position_size_percent": 2.0,
        "correlation_threshold": 0.7,
        "volatility_threshold": 15.0
    }

@pytest.fixture
def sample_portfolio():
    """Fixture that provides a sample portfolio for testing."""
    return {
        "positions": [
            {"symbol": "EUR/USD", "size": 10000, "entry_price": 1.1050},
            {"symbol": "GBP/USD", "size": 5000, "entry_price": 1.3020},
        ],
        "balance": 100000.0,
        "equity": 101200.0,
        "margin_used": 1500.0
    }
