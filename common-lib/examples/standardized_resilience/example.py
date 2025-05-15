"""
Standardized Resilience Example

This example demonstrates how to use the standardized resilience patterns in the common-lib package.
"""

import asyncio
import logging
import random
import time
from typing import Dict, Any, Optional, List

from common_lib.resilience import (
    # Core resilience components
    CircuitBreaker,
    CircuitState,
    RetryPolicy,
    Bulkhead,
    Timeout,
    
    # Standardized configuration
    StandardCircuitBreakerConfig,
    RetryConfig,
    BulkheadConfig,
    TimeoutConfig,
    StandardResilienceConfig,
    ResilienceProfiles,
    get_resilience_config,
    
    # Factory functions
    create_circuit_breaker,
    create_retry_policy,
    create_bulkhead,
    create_timeout,
    create_resilience,
    get_circuit_breaker,
    get_retry_policy,
    get_bulkhead,
    get_timeout,
    get_resilience,
    
    # Enhanced decorators
    with_standard_circuit_breaker,
    with_standard_retry,
    with_standard_bulkhead,
    with_standard_timeout,
    with_standard_resilience,
    with_database_resilience,
    with_broker_api_resilience,
    with_market_data_resilience,
    with_external_api_resilience,
    with_critical_resilience,
    with_high_throughput_resilience
)
from common_lib.errors.base_exceptions import ServiceError, ErrorCode


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("standardized-resilience-example")


# Example service client
class ExampleServiceClient:
    """Example service client that demonstrates standardized resilience patterns."""
    
    def __init__(self, service_url: str = "http://example-service:8000"):
        """
        Initialize the service client.
        
        Args:
            service_url: URL of the service
        """
        self.service_url = service_url
        self.logger = logger.getChild("ExampleServiceClient")
    
    async def _simulate_service_call(self, item_id: str, fail_probability: float = 0.3) -> Dict[str, Any]:
        """
        Simulate a service call with a chance of failure.
        
        Args:
            item_id: ID of the item to get
            fail_probability: Probability of failure (0.0-1.0)
            
        Returns:
            Simulated service response
            
        Raises:
            ServiceError: If the simulated service call fails
        """
        # Simulate network latency
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Simulate failure
        if random.random() < fail_probability:
            error_type = random.choice([
                "connection_error",
                "timeout_error",
                "server_error",
                "not_found_error"
            ])
            
            if error_type == "connection_error":
                self.logger.error(f"Connection error for item {item_id}")
                raise ServiceError(
                    message=f"Failed to connect to service for item {item_id}",
                    error_code=ErrorCode.SERVICE_DEPENDENCY_ERROR
                )
            elif error_type == "timeout_error":
                self.logger.error(f"Timeout error for item {item_id}")
                raise asyncio.TimeoutError(f"Timeout while getting item {item_id}")
            elif error_type == "server_error":
                self.logger.error(f"Server error for item {item_id}")
                raise ServiceError(
                    message=f"Server error while getting item {item_id}",
                    error_code=ErrorCode.SERVICE_UNAVAILABLE
                )
            elif error_type == "not_found_error":
                self.logger.error(f"Not found error for item {item_id}")
                raise ServiceError(
                    message=f"Item {item_id} not found",
                    error_code=ErrorCode.RESOURCE_NOT_FOUND
                )
        
        # Simulate successful response
        return {
            "id": item_id,
            "name": f"Item {item_id}",
            "description": f"Description for item {item_id}",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-02T00:00:00Z"
        }
    
    # Example method with factory-created resilience components
    async def get_data_with_factory_components(self, item_id: str) -> Dict[str, Any]:
        """
        Get data from the service with factory-created resilience components.
        
        Args:
            item_id: ID of the item to get
            
        Returns:
            Data from the service
            
        Raises:
            ServiceError: If the service call fails
        """
        # Get resilience components
        circuit_breaker = get_circuit_breaker(
            service_name="example-service",
            resource_name="get-data",
            service_type="external-api"
        )
        
        retry_policy = get_retry_policy(
            service_name="example-service",
            operation_name="get-data",
            service_type="external-api",
            exceptions=[ServiceError, asyncio.TimeoutError]
        )
        
        bulkhead = get_bulkhead(
            service_name="example-service",
            operation_name="get-data",
            service_type="external-api"
        )
        
        timeout = get_timeout(
            service_name="example-service",
            operation_name="get-data",
            service_type="external-api"
        )
        
        # Define the operation to execute
        async def operation():
            # Simulate a service call
            return await self._simulate_service_call(item_id)
        
        # Apply resilience patterns
        try:
            # Apply timeout
            async def with_timeout_applied():
                return await timeout.execute(operation)
            
            # Apply bulkhead
            async def with_bulkhead_applied():
                return await bulkhead.execute(with_timeout_applied)
            
            # Apply retry policy
            async def with_retry_applied():
                return await retry_policy.execute(with_bulkhead_applied)
            
            # Apply circuit breaker
            result = await circuit_breaker.execute(with_retry_applied)
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to get data for item {item_id}: {str(e)}")
            raise ServiceError(
                message=f"Failed to get data for item {item_id}",
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
            # Simulate a service call directly
            return await self._simulate_service_call(item_id)
        except Exception as e:
            self.logger.error(f"Failed to get data for item {item_id}: {str(e)}")
            raise ServiceError(
                message=f"Failed to get data for item {item_id}",
                error_code=ErrorCode.SERVICE_DEPENDENCY_ERROR,
                cause=e
            )
    
    # Example method with standard resilience decorator
    async def get_data_with_standard_resilience(self, item_id: str) -> Dict[str, Any]:
        """
        Get data from the service with standard resilience decorator.
        
        Args:
            item_id: ID of the item to get
            
        Returns:
            Data from the service
            
        Raises:
            ServiceError: If the service call fails
        """
        try:
            return await self._simulate_service_call(item_id)
        except Exception as e:
            self.logger.error(f"Failed to get data for item {item_id}: {str(e)}")
            raise ServiceError(
                message=f"Failed to get data for item {item_id}",
                error_code=ErrorCode.SERVICE_DEPENDENCY_ERROR,
                cause=e
            )
    
    # Example method with external API resilience decorator
    @with_external_api_resilience(
        service_name="example-service",
        operation_name="get-data-external-api"
    )
    async def get_data_with_external_api_resilience(self, item_id: str) -> Dict[str, Any]:
        """
        Get data from the service with external API resilience decorator.
        
        Args:
            item_id: ID of the item to get
            
        Returns:
            Data from the service
            
        Raises:
            ServiceError: If the service call fails
        """
        return await self._simulate_service_call(item_id)
    
    # Example method with broker API resilience decorator
    @with_broker_api_resilience(
        service_name="example-service",
        operation_name="get-data-broker-api"
    )
    async def get_data_with_broker_api_resilience(self, item_id: str) -> Dict[str, Any]:
        """
        Get data from the service with broker API resilience decorator.
        
        Args:
            item_id: ID of the item to get
            
        Returns:
            Data from the service
            
        Raises:
            ServiceError: If the service call fails
        """
        return await self._simulate_service_call(item_id)
    
    # Example method with market data resilience decorator
    @with_market_data_resilience(
        service_name="example-service",
        operation_name="get-data-market-data"
    )
    async def get_data_with_market_data_resilience(self, item_id: str) -> Dict[str, Any]:
        """
        Get data from the service with market data resilience decorator.
        
        Args:
            item_id: ID of the item to get
            
        Returns:
            Data from the service
            
        Raises:
            ServiceError: If the service call fails
        """
        return await self._simulate_service_call(item_id)
    
    # Example method with database resilience decorator
    @with_database_resilience(
        service_name="example-service",
        operation_name="get-data-database"
    )
    async def get_data_with_database_resilience(self, item_id: str) -> Dict[str, Any]:
        """
        Get data from the service with database resilience decorator.
        
        Args:
            item_id: ID of the item to get
            
        Returns:
            Data from the service
            
        Raises:
            ServiceError: If the service call fails
        """
        return await self._simulate_service_call(item_id)
    
    # Example method with critical resilience decorator
    @with_critical_resilience(
        service_name="example-service",
        operation_name="get-data-critical"
    )
    async def get_data_with_critical_resilience(self, item_id: str) -> Dict[str, Any]:
        """
        Get data from the service with critical resilience decorator.
        
        Args:
            item_id: ID of the item to get
            
        Returns:
            Data from the service
            
        Raises:
            ServiceError: If the service call fails
        """
        return await self._simulate_service_call(item_id)
    
    # Example method with high throughput resilience decorator
    @with_high_throughput_resilience(
        service_name="example-service",
        operation_name="get-data-high-throughput"
    )
    async def get_data_with_high_throughput_resilience(self, item_id: str) -> Dict[str, Any]:
        """
        Get data from the service with high throughput resilience decorator.
        
        Args:
            item_id: ID of the item to get
            
        Returns:
            Data from the service
            
        Raises:
            ServiceError: If the service call fails
        """
        return await self._simulate_service_call(item_id)
    
    # Example method with individual standard resilience decorators
    async def get_data_with_individual_decorators(self, item_id: str) -> Dict[str, Any]:
        """
        Get data from the service with individual standard resilience decorators.
        
        Args:
            item_id: ID of the item to get
            
        Returns:
            Data from the service
            
        Raises:
            ServiceError: If the service call fails
        """
        try:
            return await self._simulate_service_call(item_id)
        except Exception as e:
            self.logger.error(f"Failed to get data for item {item_id}: {str(e)}")
            raise ServiceError(
                message=f"Failed to get data for item {item_id}",
                error_code=ErrorCode.SERVICE_DEPENDENCY_ERROR,
                cause=e
            )


# Example usage
async def main():
    """Run the example."""
    client = ExampleServiceClient()
    
    # Example with factory-created resilience components
    logger.info("Testing factory-created resilience components")
    try:
        result = await client.get_data_with_factory_components("item1")
        logger.info(f"Result with factory components: {result}")
    except Exception as e:
        logger.error(f"Error with factory components: {str(e)}")
    
    # Example with combined resilience
    logger.info("\nTesting combined resilience")
    try:
        result = await client.get_data_with_combined_resilience("item2")
        logger.info(f"Result with combined resilience: {result}")
    except Exception as e:
        logger.error(f"Error with combined resilience: {str(e)}")
    
    # Example with standard resilience decorator
    logger.info("\nTesting standard resilience decorator")
    try:
        result = await client.get_data_with_standard_resilience("item3")
        logger.info(f"Result with standard resilience decorator: {result}")
    except Exception as e:
        logger.error(f"Error with standard resilience decorator: {str(e)}")
    
    # Example with external API resilience decorator
    logger.info("\nTesting external API resilience decorator")
    try:
        result = await client.get_data_with_external_api_resilience("item4")
        logger.info(f"Result with external API resilience decorator: {result}")
    except Exception as e:
        logger.error(f"Error with external API resilience decorator: {str(e)}")
    
    # Example with broker API resilience decorator
    logger.info("\nTesting broker API resilience decorator")
    try:
        result = await client.get_data_with_broker_api_resilience("item5")
        logger.info(f"Result with broker API resilience decorator: {result}")
    except Exception as e:
        logger.error(f"Error with broker API resilience decorator: {str(e)}")
    
    # Example with market data resilience decorator
    logger.info("\nTesting market data resilience decorator")
    try:
        result = await client.get_data_with_market_data_resilience("item6")
        logger.info(f"Result with market data resilience decorator: {result}")
    except Exception as e:
        logger.error(f"Error with market data resilience decorator: {str(e)}")
    
    # Example with database resilience decorator
    logger.info("\nTesting database resilience decorator")
    try:
        result = await client.get_data_with_database_resilience("item7")
        logger.info(f"Result with database resilience decorator: {result}")
    except Exception as e:
        logger.error(f"Error with database resilience decorator: {str(e)}")
    
    # Example with critical resilience decorator
    logger.info("\nTesting critical resilience decorator")
    try:
        result = await client.get_data_with_critical_resilience("item8")
        logger.info(f"Result with critical resilience decorator: {result}")
    except Exception as e:
        logger.error(f"Error with critical resilience decorator: {str(e)}")
    
    # Example with high throughput resilience decorator
    logger.info("\nTesting high throughput resilience decorator")
    try:
        result = await client.get_data_with_high_throughput_resilience("item9")
        logger.info(f"Result with high throughput resilience decorator: {result}")
    except Exception as e:
        logger.error(f"Error with high throughput resilience decorator: {str(e)}")
    
    # Example with individual standard resilience decorators
    logger.info("\nTesting individual standard resilience decorators")
    try:
        result = await client.get_data_with_individual_decorators("item10")
        logger.info(f"Result with individual standard resilience decorators: {result}")
    except Exception as e:
        logger.error(f"Error with individual standard resilience decorators: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())