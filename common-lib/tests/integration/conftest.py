"""
Pytest configuration for integration tests.
"""

import os
import sys
import pytest
import asyncio
from pathlib import Path

# Add parent directory to path to ensure imports work correctly
parent_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(parent_dir))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()