
"""
Apply Caching to Services (Simple Version)

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

def find_read_repositories(service_dir):
    """
    Find all read repositories in a service directory.
    
    Args:
        service_dir: The service directory to search in
        
    Returns:
        A list of read repository file paths
    """
    read_repos = []
    
    # Walk through the service directory
    for root, dirs, files in os.walk(service_dir):
        # Skip __pycache__ directories
        if "__pycache__" in root:
            continue
            
        # Check each Python file
        for file in files:
            if not file.endswith(".py"):
                continue
                
            # Check if the file is a read repository
            if "read_repository" in file.lower() or "readrepository" in file.lower():
                read_repos.append(os.path.join(root, file))
    
    return read_repos

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
        
        # Find read repositories
        read_repos = find_read_repositories(service_dir)
        
        if not read_repos:
            logger.warning(f"No read repositories found in {service_name}")
            continue
        
        logger.info(f"Found {len(read_repos)} read repositories in {service_name}")
        for repo_path in read_repos:
            logger.info(f"  - {repo_path}")
    
    logger.info("Finished applying caching to services")

if __name__ == "__main__":
    main()
