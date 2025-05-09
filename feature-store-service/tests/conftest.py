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

# Import fixtures from other test modules
pytest_plugins = [
    "tests.fixtures.reliability_fixtures",
]
