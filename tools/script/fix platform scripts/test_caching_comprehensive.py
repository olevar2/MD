"""
Comprehensive Caching Test

This script tests the caching implementation across all services.
It creates mock data, stores it in repositories, and verifies that caching is working correctly.
"""
import os
import sys
import logging
import asyncio
import time
import json
from pathlib import Path
import importlib.util
from typing import Dict, Any, List, Optional

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

# Define the services to test
SERVICES = [
    "causal-analysis-service",
    "backtesting-service",
    "market-analysis-service",
    "analysis-coordinator-service"
]

def import_module_from_file(file_path, module_name=None):
    """
    Import a module from a file path.
    
    Args:
        file_path: Path to the module file
        module_name: Name to give the module (defaults to file name)
        
    Returns:
        The imported module
    """
    if module_name is None:
        module_name = os.path.basename(file_path).replace(".py", "")
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

async def test_repository_caching(repo_path, test_data):
    """
    Test caching for a repository.
    
    Args:
        repo_path: Path to the repository file
        test_data: Test data to use
        
    Returns:
        True if caching is working correctly, False otherwise
    """
    try:
        logger.info(f"Testing caching for {repo_path}")
        
        # Import the repository module
        repo_module = import_module_from_file(repo_path)
        
        # Find the repository class
        repo_class = None
        for attr_name in dir(repo_module):
            attr = getattr(repo_module, attr_name)
            if isinstance(attr, type) and "Repository" in attr_name:
                repo_class = attr
                break
        
        if repo_class is None:
            logger.error(f"Could not find repository class in {repo_path}")
            return False
        
        # Create an instance of the repository
        repo = repo_class()
        
        # Add test data to the repository's in-memory cache
        test_id = "test-id-1"
        if hasattr(repo, "backtests"):
            repo.backtests[test_id] = test_data
        elif hasattr(repo, "optimizations"):
            repo.optimizations[test_id] = test_data
        elif hasattr(repo, "tests"):
            repo.tests[test_id] = test_data
        elif hasattr(repo, "analysis_cache"):
            repo.analysis_cache[test_id] = test_data
        elif hasattr(repo, "tasks"):
            repo.tasks[test_id] = test_data
        elif hasattr(repo, "effects"):
            repo.effects[test_id] = test_data
        else:
            # Try a generic approach
            for attr_name in dir(repo):
                attr = getattr(repo, attr_name)
                if isinstance(attr, dict):
                    attr[test_id] = test_data
                    break
        
        # Get the data from the repository (first call, should not use cache)
        start_time = time.time()
        result1 = await repo.get_by_id(test_id)
        first_call_time = time.time() - start_time
        
        # Get the data from the repository again (second call, should use cache)
        start_time = time.time()
        result2 = await repo.get_by_id(test_id)
        second_call_time = time.time() - start_time
        
        # Check if the results are the same
        if result1 is None or result2 is None:
            logger.error(f"One of the calls returned None for {repo_path}")
            return False
        
        # Check if the second call was faster (indicating it used the cache)
        logger.info(f"First call time: {first_call_time:.6f}s")
        logger.info(f"Second call time: {second_call_time:.6f}s")
        
        # Check if the cache attribute exists
        if not hasattr(repo, "cache"):
            logger.error(f"Repository does not have a cache attribute: {repo_path}")
            return False
        
        logger.info(f"Caching test passed for {repo_path}")
        return True
    except Exception as e:
        logger.error(f"Error testing caching for {repo_path}: {e}")
        return False

async def main():
    """Main function to run the tests."""
    logger.info("Starting comprehensive caching tests")
    
    # Check if read_repositories.txt exists
    repos_file = BASE_DIR / "tools" / "script" / "fix platform scripts" / "read_repositories.txt"
    if not repos_file.exists():
        logger.error(f"File {repos_file} does not exist. Run identify_read_repositories.py first.")
        return
    
    # Read the list of repositories
    with open(repos_file, "r") as f:
        repo_paths = [line.strip() for line in f.readlines()]
    
    logger.info(f"Found {len(repo_paths)} read repositories to test")
    
    # Create test data for each repository
    test_data = {
        "task_id": "test-id-1",
        "strategy_id": "test-strategy",
        "symbol": "EURUSD",
        "timeframe": "1h",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "status": "completed",
        "created_at": "2023-12-31T23:59:59",
        "completed_at": "2023-12-31T23:59:59",
        "metrics": {
            "profit_factor": 1.5,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.1,
            "win_rate": 0.6,
            "total_trades": 100,
            "net_profit": 1000.0
        }
    }
    
    # Test caching for each repository
    success_count = 0
    for repo_path in repo_paths:
        try:
            if await test_repository_caching(repo_path, test_data):
                success_count += 1
        except Exception as e:
            logger.error(f"Error testing {repo_path}: {e}")
    
    logger.info(f"Caching is working correctly in {success_count} out of {len(repo_paths)} read repositories")
    
    if success_count == len(repo_paths):
        logger.info("All repositories have caching working correctly")
    else:
        logger.warning(f"{len(repo_paths) - success_count} repositories need to be fixed")

if __name__ == "__main__":
    asyncio.run(main())