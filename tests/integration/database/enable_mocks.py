"""
Script to enable mocks for database integration tests.

This script sets the USE_MOCKS flag to True in the database config module.
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add common-lib to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'common-lib'))

# Import and set USE_MOCKS flag
from common_lib.database.config import USE_MOCKS
import common_lib.database.config

# Set USE_MOCKS to True
common_lib.database.config.USE_MOCKS = True

logger.info("Mocks enabled for database integration tests")

# Print current value of USE_MOCKS
logger.info(f"USE_MOCKS = {common_lib.database.config.USE_MOCKS}")

# Verify that the flag is set correctly
if common_lib.database.config.USE_MOCKS:
    logger.info("Mocks are enabled")
else:
    logger.error("Failed to enable mocks")
    sys.exit(1)

logger.info("Database mocks enabled successfully")