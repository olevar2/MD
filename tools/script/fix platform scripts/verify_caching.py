"""
Verify Caching Implementation

This script verifies that caching has been implemented correctly in all read repositories.
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

def verify_repository_caching(repo_path):
    """
    Verify that caching has been implemented correctly in a repository.
    
    Args:
        repo_path: Path to the repository file
        
    Returns:
        True if caching is implemented correctly, False otherwise
    """
    try:
        with open(repo_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check for cache import
        has_cache_import = "from common_lib.caching.decorators import cached" in content
        
        # Check for cache factory import
        has_cache_factory_import = "cache_factory" in content
        
        # Check for cache initialization in __init__
        has_cache_init = re.search(r"def\s+__init__.*\n.*\n.*self\.cache\s*=\s*cache_factory\.get_cache\(\)", content, re.DOTALL) is not None
        
        # Check for cached decorator on get methods
        has_cached_decorator = re.search(r"@cached\(cache_factory\.get_cache\(\),\s*['\"].*['\"],\s*ttl=\d+\)", content) is not None
        
        # Check if all requirements are met
        is_correctly_implemented = has_cache_import and has_cache_factory_import and has_cache_init and has_cached_decorator
        
        if is_correctly_implemented:
            logger.info(f"Caching is correctly implemented in {repo_path}")
        else:
            logger.warning(f"Caching is not correctly implemented in {repo_path}")
            if not has_cache_import:
                logger.warning(f"  - Missing cache import")
            if not has_cache_factory_import:
                logger.warning(f"  - Missing cache factory import")
            if not has_cache_init:
                logger.warning(f"  - Missing cache initialization in __init__")
            if not has_cached_decorator:
                logger.warning(f"  - Missing cached decorator on get methods")
        
        return is_correctly_implemented
    except Exception as e:
        logger.error(f"Error verifying caching in {repo_path}: {e}")
        return False

def main():
    """Main function to verify caching implementation."""
    logger.info("Starting to verify caching implementation")
    
    # Check if read_repositories.txt exists
    repos_file = BASE_DIR / "tools" / "script" / "fix platform scripts" / "read_repositories.txt"
    if not repos_file.exists():
        logger.error(f"File {repos_file} does not exist. Run identify_read_repositories.py first.")
        return
    
    # Read the list of repositories
    with open(repos_file, "r") as f:
        repo_paths = [line.strip() for line in f.readlines()]
    
    logger.info(f"Found {len(repo_paths)} read repositories to verify")
    
    # Verify caching in each repository
    correct_count = 0
    for repo_path in repo_paths:
        if verify_repository_caching(repo_path):
            correct_count += 1
    
    logger.info(f"Caching is correctly implemented in {correct_count} out of {len(repo_paths)} read repositories")
    
    # Update the platform_fixing_log2.md if all repositories are correctly implemented
    if correct_count == len(repo_paths):
        logger.info("All repositories have caching correctly implemented")
    else:
        logger.warning(f"{len(repo_paths) - correct_count} repositories need to be fixed")

if __name__ == "__main__":
    main()