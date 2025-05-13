"""
Resilience Example

This example demonstrates how to use the resilience patterns in the common-lib package.
"""

import asyncio
import logging
import random
import time
from typing import Dict, Any, Optional, List

from common_lib.resilience import (
    CircuitBreaker,
    CircuitState,
    retry,
    RetryPolicy,
    Bulkhead,
    Timeout,
    resilient,
    ResilienceConfig,
    Resilience
)
from common_lib.errors.base_exceptions import ServiceError, ErrorCode


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("resilience-example")


# Example service client
class ExampleServiceClient:
    """Example service client that demonstrates resilience patterns."""
    
    def __init__(self, service_url: str = "http://example-service:8000"):
        """
        Initialize the service client.
        
        Args:
            service_url: URL of the service
        """
        self.service_url = service_url
        self.logger = logger.getChild("ExampleServiceClient")
        
        # Create resilience components
        self.circuit_breaker = CircuitBreaker(
            "example-service",
            logger=self.logger
        )
        
        self.retry_policy = RetryPolicy(
            retries=3,
            delay=1.0,
            max_delay=5.0,
            backoff=2.0,
            logger=self.logger
        )
        
        self.bulkhead = Bulkhead(
            "example-service",
            max_concurrent_calls=5,
            max_queue_size=10,
            logger=self.logger
        )
        
        self.timeout = Timeout(
            5.0,
            operation="example-service",
            logger=self.logger
        )
        
        # Create combined resilience
        self.resilience_config = ResilienceConfig(
            service_name="example-service",
            operation_name="get-data",
            enable_circuit_breaker=True,
            failure_threshold=3,
            recovery_timeout=10.0,
            enable_retry=True,
            max_retries=3,
            retry_delay=1.0,
            max_delay=5.0,
            backoff=2.0,
            enable_bulkhead=True,
            max_concurrent_calls=5,
            max_queue_size=10,
            enable_timeout=True,
            timeout=5.0
        )
        self.resilience = Resilience(self.resilience_config, logger=self.logger)
    
    # Example method with individual resilience patterns
    async def get_data_with_individual_patterns(self, item_id: str) -> Dict[str, Any]:
        """
        Get data from the service with individual resilience patterns.
        
        Args:
            item_id: ID of the item to get
            
        Returns:
            Data from the service
            
        Raises:
            ServiceError: If the service call fails
        """
        # Define the operation to execute
        async def operation():
            # Simulate a service call
            return await self._simulate_service_call(item_id)
        
        # Apply resilience patterns
        try:
            # Apply timeout
            async def with_timeout():
                return await self.timeout.execute(operation)
            
            # Apply bulkhead
            async def with_bulkhead():
                return await self.bulkhead.execute(with_timeout)
            
            # Apply circuit breaker
            async def with_circuit_breaker():
                return await self.circuit_breaker.execute(with_bulkhead)
            
            # Apply retry
            return await self.retry_policy.execute(with_circuit_breaker)
        except Exception as e:
            self.logger.error(f"Failed to get data for item {item_id}: {str(e)}")
            raise ServiceError(
                f"Failed to get data for item {item_id}",
                error_code=ErrorCode.SERVICE_DEPENDENCY_ERROR,
                cause=e
            )
    
    # Example method with combined resilience
    async def get_data_with_combined_resilience(self, item_id: str) -> Dict[str, Any]:
        """
        Get data from the service with combined resilience.
        
        Args:
            item_id: ID of the item to get
            
        Returns:
            Data from the service
            
        Raises:
            ServiceError: If the service call fails
        """
        try:
            return await self.resilience.execute_async(self._simulate_service_call, item_id)
        except Exception as e:
            self.logger.error(f"Failed to get data for item {item_id}: {str(e)}")
            raise ServiceError(
                f"Failed to get data for item {item_id}",
                error_code=ErrorCode.SERVICE_DEPENDENCY_ERROR,
                cause=e
            )
    
    # Example method with resilient decorator
    @resilient(
        service_name="example-service",
        operation_name="get-data-decorator",
        enable_circuit_breaker=True,
        failure_threshold=3,
        recovery_timeout=10.0,
        enable_retry=True,
        max_retries=3,
        retry_delay=1.0,
        max_delay=5.0,
        backoff=2.0,
        enable_bulkhead=True,
        max_concurrent_calls=5,
        max_queue_size=10,
        enable_timeout=True,
        timeout=5.0
    )
    async def get_data_with_decorator(self, item_id: str) -> Dict[str, Any]:
        """
        Get data from the service with resilient decorator.
        
        Args:
            item_id: ID of the item to get
            
        Returns:
            Data from the service
            
        Raises:
            ServiceError: If the service call fails
        """
        return await self._simulate_service_call(item_id)
    
    # Simulate a service call with random failures
    async def _simulate_service_call(self, item_id: str) -> Dict[str, Any]:
        """
        Simulate a service call with random failures.
        
        Args:
            item_id: ID of the item to get
            
        Returns:
            Simulated data
            
        Raises:
            ServiceError: If the service call fails
            TimeoutError: If the service call times out
        """
        # Simulate random failures
        failure_type = random.randint(0, 10)
        
        if failure_type == 0:
            # Simulate a timeout
            self.logger.info(f"Simulating a timeout for item {item_id}")
            await asyncio.sleep(10.0)
            return {"item_id": item_id, "data": "timeout data"}
        elif failure_type == 1:
            # Simulate a service error
            self.logger.info(f"Simulating a service error for item {item_id}")
            raise ServiceError(
                f"Service error for item {item_id}",
                error_code=ErrorCode.SERVICE_UNAVAILABLE
            )
        elif failure_type == 2:
            # Simulate a validation error
            self.logger.info(f"Simulating a validation error for item {item_id}")
            raise ServiceError(
                f"Validation error for item {item_id}",
                error_code=ErrorCode.VALIDATION_ERROR
            )
        else:
            # Simulate a successful call
            self.logger.info(f"Simulating a successful call for item {item_id}")
            await asyncio.sleep(0.1)  # Simulate some processing time
            return {"item_id": item_id, "data": f"data for {item_id}"}


# Example usage
async def main():
    """Run the example."""
    client = ExampleServiceClient()
    
    # Example with individual resilience patterns
    logger.info("Testing individual resilience patterns")
    try:
        result = await client.get_data_with_individual_patterns("item1")
        logger.info(f"Result with individual patterns: {result}")
    except Exception as e:
        logger.error(f"Error with individual patterns: {str(e)}")
    
    # Example with combined resilience
    logger.info("\nTesting combined resilience")
    try:
        result = await client.get_data_with_combined_resilience("item2")
        logger.info(f"Result with combined resilience: {result}")
    except Exception as e:
        logger.error(f"Error with combined resilience: {str(e)}")
    
    # Example with resilient decorator
    logger.info("\nTesting resilient decorator")
    try:
        result = await client.get_data_with_decorator("item3")
        logger.info(f"Result with resilient decorator: {result}")
    except Exception as e:
        logger.error(f"Error with resilient decorator: {str(e)}")
    
    # Test multiple concurrent calls
    logger.info("\nTesting multiple concurrent calls")
    tasks = []
    for i in range(20):
        tasks.append(client.get_data_with_combined_resilience(f"item{i+4}"))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    success_count = 0
    error_count = 0
    for result in results:
        if isinstance(result, Exception):
            error_count += 1
        else:
            success_count += 1
    
    logger.info(f"Multiple concurrent calls: {success_count} succeeded, {error_count} failed")


if __name__ == "__main__":
    asyncio.run(main())
