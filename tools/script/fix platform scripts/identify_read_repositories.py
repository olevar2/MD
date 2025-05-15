"""
Identify Read Repositories

This script identifies all read repositories in the forex trading platform.
"""
import os
import sys
import logging
from pathlib import Path
import re

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

# Define the services to check
SERVICES = [
    "causal-analysis-service",
    "backtesting-service",
    "market-analysis-service",
    "analysis-coordinator-service"
]

def find_read_repositories(service_dir):
    """
    Find all read repositories in a service.
    
    Args:
        service_dir: The service directory
        
    Returns:
        A list of read repository file paths
    """
    read_repositories = []
    
    # Check for read_repositories directory
    read_repos_dir = os.path.join(service_dir, "repositories", "read_repositories")
    if os.path.exists(read_repos_dir):
        for filename in os.listdir(read_repos_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                read_repositories.append(os.path.join(read_repos_dir, filename))
    
    # Check for files with "read_repository" in the name
    for root, _, files in os.walk(service_dir):
        for filename in files:
            if filename.endswith(".py") and "read_repository" in filename.lower():
                file_path = os.path.join(root, filename)
                if file_path not in read_repositories:
                    read_repositories.append(file_path)
    
    # Check for files that inherit from ReadRepository
    for root, _, files in os.walk(service_dir):
        for filename in files:
            if filename.endswith(".py"):
                file_path = os.path.join(root, filename)
                if file_path not in read_repositories:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            if re.search(r"class\s+\w+\s*\(\s*.*ReadRepository", content):
                                read_repositories.append(file_path)
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {e}")
    
    return read_repositories

def main():
    """Main function to identify read repositories."""
    logger.info("Starting to identify read repositories")
    
    all_read_repositories = []
    
    for service_name in SERVICES:
        service_dir = BASE_DIR / service_name
        
        if not service_dir.exists():
            logger.warning(f"Service directory {service_dir} does not exist")
            continue
        
        logger.info(f"Processing service: {service_name}")
        
        # Find read repositories
        read_repositories = find_read_repositories(service_dir)
        
        if read_repositories:
            logger.info(f"Found {len(read_repositories)} read repositories in {service_name}")
            for repo in read_repositories:
                logger.info(f"  - {repo}")
                all_read_repositories.append(repo)
        else:
            logger.info(f"No read repositories found in {service_name}")
    
    logger.info(f"Found a total of {len(all_read_repositories)} read repositories")
    
    # Write results to a file
    output_file = BASE_DIR / "tools" / "script" / "fix platform scripts" / "read_repositories.txt"
    with open(output_file, "w") as f:
        for repo in all_read_repositories:
            f.write(f"{repo}\n")
    
    logger.info(f"Wrote results to {output_file}")

if __name__ == "__main__":
    main()