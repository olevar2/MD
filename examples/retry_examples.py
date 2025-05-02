"""
Retry Logic Examples

This file demonstrates various ways to use the centralized retry logic
from common_lib.resilience.

Requirements:
- common_lib needs to be in your Python path
- Requires core_foundations to be installed or in Python path

Run with:
python -m examples.retry_examples
"""

import asyncio
import random
import logging
from typing import Dict, Any, List, Optional

from common_lib.resilience import (
    retry_with_policy,
    RetryExhaustedException,
    register_common_retryable_exceptions,
    register_database_retryable_exceptions
)

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Example 1: Basic retry with default settings
@retry_with_policy(
    max_attempts=3,
    base_delay=1.0,
    exceptions=[ConnectionError, TimeoutError]
)
def fetch_data(success_rate: float = 0.3) -> str:
    """
    A function that simulates network requests with occasional failures.
    
    Args:
        success_rate: Probability of success (0.0 to 1.0)
        
    Returns:
        str: A sample data response
        
    Raises:
        ConnectionError: Randomly to simulate network issues
        TimeoutError: Randomly to simulate timeouts
    """
    logger.info("Attempting to fetch data...")
    
    if random.random() > success_rate:
        error_type = random.choice([ConnectionError, TimeoutError])
        logger.warning(f"Simulating a {error_type.__name__}")
        raise error_type("Simulated failure")
        
    return "Sample data response"


# Example 2: Async function with retry and monitoring
def mock_metric_handler(metric_name: str, metric_data: Dict[str, Any]) -> None:
    """Mock function to simulate recording metrics."""
    logger.info(f"Recording metric: {metric_name}")
    logger.info(f"Metric data: {metric_data}")


@retry_with_policy(
    max_attempts=4,
    base_delay=0.5,
    max_delay=5.0,
    backoff_factor=2.0,
    jitter=True,
    exceptions=[ConnectionError, TimeoutError],
    metric_handler=mock_metric_handler,
    service_name="example-service",
    operation_name="fetch_user_async"
)
async def fetch_user_async(user_id: str, success_rate: float = 0.4) -> Dict[str, Any]:
    """
    Asynchronously fetch user data with retry capability.
    
    Args:
        user_id: The ID of the user to fetch
        success_rate: Probability of success (0.0 to 1.0)
        
    Returns:
        Dict[str, Any]: User data
        
    Raises:
        ConnectionError: Randomly to simulate network issues
        TimeoutError: Randomly to simulate timeouts
    """
    logger.info(f"Attempting to fetch user {user_id}...")
    
    # Simulate network delay
    await asyncio.sleep(0.2)
    
    if random.random() > success_rate:
        error_type = random.choice([ConnectionError, TimeoutError])
        logger.warning(f"Simulating a {error_type.__name__}")
        raise error_type(f"Failed to fetch user {user_id}")
        
    return {"id": user_id, "name": f"User {user_id}", "status": "active"}


# Example 3: Class-based configuration
class ApiClient:
    """Example API client with configurable retry settings."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the API client with retry settings.
        
        Args:
            config: Optional configuration dictionary
        """
        config = config or {}
        self.max_attempts = config.get("max_retries", 3)
        self.base_delay = config.get("base_delay", 1.0)
        self.max_delay = config.get("max_delay", 30.0)
        self.backoff_factor = config.get("backoff_factor", 2.0)
        self.retryable_exceptions = [ConnectionError, TimeoutError]
        
    @retry_with_policy(
        exceptions=lambda self: self.retryable_exceptions,
        max_attempts=lambda self: self.max_attempts,
        base_delay=lambda self: self.base_delay,
        max_delay=lambda self: self.max_delay,
        backoff_factor=lambda self: self.backoff_factor,
        service_name="api-client",
        operation_name="get_resource"
    )
    def get_resource(self, resource_id: str, success_rate: float = 0.5) -> Dict[str, Any]:
        """
        Get a resource by ID with retry capability.
        
        Args:
            resource_id: The ID of the resource to fetch
            success_rate: Probability of success (0.0 to 1.0)
            
        Returns:
            Dict[str, Any]: Resource data
            
        Raises:
            ConnectionError: Randomly to simulate network issues
            TimeoutError: Randomly to simulate timeouts
        """
        logger.info(f"Attempting to fetch resource {resource_id}...")
        
        if random.random() > success_rate:
            error_type = random.choice([ConnectionError, TimeoutError])
            logger.warning(f"Simulating a {error_type.__name__}")
            raise error_type(f"Failed to fetch resource {resource_id}")
            
        return {"id": resource_id, "type": "example", "status": "active"}


# Example 4: Using pre-registered exceptions
@retry_with_policy(
    max_attempts=3,
    base_delay=0.5,
    exceptions=register_common_retryable_exceptions(),
    service_name="example-service",
    operation_name="network_operation"
)
def network_operation(success_rate: float = 0.4) -> str:
    """
    Perform a network operation with common retryable exceptions.
    
    Args:
        success_rate: Probability of success (0.0 to 1.0)
        
    Returns:
        str: Operation result
    """
    logger.info("Performing network operation...")
    
    # Use common network errors that should be part of register_common_retryable_exceptions
    if random.random() > success_rate:
        # ConnectionResetError should be in common retryable exceptions
        logger.warning("Simulating ConnectionResetError")
        raise ConnectionResetError("Connection was reset")
        
    return "Network operation successful"


# Example 5: Database operation with specialized exception handling
@retry_with_policy(
    max_attempts=5,
    base_delay=0.5,
    exceptions=register_database_retryable_exceptions(),
    service_name="database-service",
    operation_name="db_operation"
)
def db_operation(success_rate: float = 0.6) -> List[Dict[str, Any]]:
    """
    Perform a database operation with database-specific retryable exceptions.
    
    Args:
        success_rate: Probability of success (0.0 to 1.0)
        
    Returns:
        List[Dict[str, Any]]: Database query results
        
    Raises:
        Various database exceptions that should be retried
    """
    logger.info("Performing database operation...")
    
    if random.random() > success_rate:
        # Simulate a database operational error
        # This is just for demo - we're using a general Exception with a name
        # that looks like a database error since we may not have the actual
        # database libraries imported in this example
        logger.warning("Simulating database connection error")
        e = Exception("Database connection lost")
        e.__class__.__name__ = "OperationalError"  # Simulate a DB error
        raise e
        
    return [{"id": 1, "value": "Sample"}, {"id": 2, "value": "Data"}]


async def main():
    """Run examples of different retry patterns."""
    logger.info("=== Starting Retry Examples ===")
    
    # Example 1: Basic retry
    logger.info("\n=== Example 1: Basic Retry ===")
    try:
        result = fetch_data(success_rate=0.3)
        logger.info(f"Success: {result}")
    except RetryExhaustedException as e:
        logger.error(f"All retries failed: {e}")
    
    # Example 2: Async retry
    logger.info("\n=== Example 2: Async Retry ===")
    try:
        user = await fetch_user_async("user123", success_rate=0.4)
        logger.info(f"User fetched: {user}")
    except RetryExhaustedException as e:
        logger.error(f"All retries failed: {e}")
    
    # Example 3: Class-based configuration
    logger.info("\n=== Example 3: Class-based Configuration ===")
    client = ApiClient({"max_retries": 4, "base_delay": 0.5})
    try:
        resource = client.get_resource("res456", success_rate=0.5)
        logger.info(f"Resource fetched: {resource}")
    except RetryExhaustedException as e:
        logger.error(f"All retries failed: {e}")
    
    # Example 4: Pre-registered exceptions
    logger.info("\n=== Example 4: Pre-registered Exceptions ===")
    try:
        result = network_operation(success_rate=0.4)
        logger.info(f"Result: {result}")
    except RetryExhaustedException as e:
        logger.error(f"All retries failed: {e}")
    
    # Example 5: Database retry
    logger.info("\n=== Example 5: Database Retry ===")
    try:
        data = db_operation(success_rate=0.6)
        logger.info(f"DB operation result: {data}")
    except RetryExhaustedException as e:
        logger.error(f"All retries failed: {e}")
    
    logger.info("=== All Examples Completed ===")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
