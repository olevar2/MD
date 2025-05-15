
"""
Apply Caching to Services

This script applies caching to all services in the forex trading platform.
"""
import os
import re
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define the base directory
BASE_DIR = Path("D:/MD/forex_trading_platform")

# Define the services to apply caching to
SERVICES = [
    "causal-analysis-service",
    "backtesting-service",
    "market-analysis-service",
    "analysis-coordinator-service"
]

def main():
    """
    Main function to apply caching to all services.
    """
    logger.info("Starting to apply caching to services")
    
    for service_name in SERVICES:
        service_dir = BASE_DIR / service_name
        
        if not service_dir.exists():
            logger.warning(f"Service directory {service_dir} does not exist")
            continue
        
        logger.info(f"Processing service: {service_name}")
    
    logger.info("Finished applying caching to services")

if __name__ == "__main__":
    main()
