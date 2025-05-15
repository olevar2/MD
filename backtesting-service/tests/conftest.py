"""
Pytest configuration file for backtesting service tests.
"""
import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import pytest plugins
pytest_plugins = ["pytest_asyncio"]
