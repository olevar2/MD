"""
Test Caching Implementation

This script tests the caching implementation for read repositories.
"""
import os
import sys
import logging
import asyncio
import time
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

# Add the necessary directories to the Python path
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "backtesting-service"))

# Import the necessary modules
from repositories.read_repositories.backtest_read_repository import BacktestReadRepository
from backtesting_service.utils.cache_factory import cache_factory as backtest_cache_factory
from backtesting_service.models.backtest_models import BacktestResult

async def test_backtest_repository_caching():
    """Test caching for the backtest repository."""
    logger.info("Testing caching for backtest repository")
    
    # Create a repository
    repo = BacktestReadRepository()
    
    # Create a test backtest result
    test_id = "test-backtest-1"
    test_backtest = {
        "task_id": test_id,
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
        },
        "trades": []
    }
    
    # Add the test backtest to the repository
    repo.backtests[test_id] = BacktestResult(**test_backtest)
    
    # Get the backtest from the repository (first call, should not use cache)
    start_time = time.time()
    result1 = await repo.get_by_id(test_id)
    first_call_time = time.time() - start_time
    
    # Get the backtest from the repository again (second call, should use cache)
    start_time = time.time()
    result2 = await repo.get_by_id(test_id)
    second_call_time = time.time() - start_time
    
    # Check if the results are the same
    assert result1 is not None, "First call returned None"
    assert result2 is not None, "Second call returned None"
    assert result1.task_id == result2.task_id, "Task IDs don't match"
    
    # Check if the second call was faster (indicating it used the cache)
    logger.info(f"First call time: {first_call_time:.6f}s")
    logger.info(f"Second call time: {second_call_time:.6f}s")
    
    # Check the cache directly
    cache = backtest_cache_factory.get_cache()
    cache_key = f"backtest:{test_id}"
    cached_value = await cache.get(cache_key)
    
    assert cached_value is not None, "Value not found in cache"
    
    logger.info("Backtest repository caching test passed")

async def main():
    """Main function to run the tests."""
    logger.info("Starting caching tests")
    
    try:
        await test_backtest_repository_caching()
        logger.info("All caching tests passed")
    except Exception as e:
        logger.error(f"Error in caching tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())