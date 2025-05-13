"""
Example implementation showing how to use the centralized resilience patterns.
"""

import asyncio
import logging
import random
import time
from typing import Dict, Any, List, Optional

from common_lib.resilience import (
    # Circuit breaker
    CircuitBreaker, CircuitBreakerConfig, CircuitState, create_circuit_breaker,
    
    # Retry policy
    retry_with_policy, register_common_retryable_exceptions,
    
    # Timeout handler
    timeout_handler,
    
    # Bulkhead
    bulkhead
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Example 1: Using retry_with_policy decorator
@retry_with_policy(
    max_attempts=3,
    base_delay=1.0,
    jitter=True,
    service_name="example-service",
    operation_name="fetch_market_data"
)
async def fetch_market_data(symbol: str) -> Dict[str, Any]:
    """
    Example function that fetches market data with retry logic.
    
    Args:
        symbol: The market symbol to fetch
        
    Returns:
        Market data dictionary
    """
    logger.info(f"Fetching market data for {symbol}")
    
    # Simulate random failures
    if random.random() < 0.7:  # 70% chance of failure on first attempt
        logger.error(f"Simulated failure fetching market data for {symbol}")
        raise ConnectionError("Simulated connection error")
        
    # Simulate successful response
    await asyncio.sleep(0.1)  # Simulate network call
    return {
        "symbol": symbol,
        "price": random.uniform(1.0, 100.0),
        "timestamp": time.time()
    }


# Example 2: Using circuit breaker
async def execute_trade_with_circuit_breaker(order_id: str, symbol: str, quantity: float) -> Dict[str, Any]:
    """
    Example function that executes a trade with circuit breaker protection.
    
    Args:
        order_id: The order ID
        symbol: The market symbol
        quantity: The quantity to trade
        
    Returns:
        Trade result dictionary
    """
    # Create a circuit breaker
    cb_config = CircuitBreakerConfig(
        failure_threshold=3,
        reset_timeout_seconds=30,
        half_open_max_calls=1
    )
    
    cb = create_circuit_breaker(
        service_name="example-service",
        resource_name="trade-execution",
        config=cb_config
    )
    
    # Define the function that is protected by the circuit breaker
    async def execute_trade() -> Dict[str, Any]:
        logger.info(f"Executing trade for order {order_id}: {quantity} {symbol}")
        
        # Simulate random failures
        if random.random() < 0.4:  # 40% chance of failure
            logger.error(f"Simulated failure executing trade for order {order_id}")
            raise ConnectionError("Simulated broker connection error")
            
        # Simulate successful response
        await asyncio.sleep(0.2)  # Simulate network call
        return {
            "order_id": order_id,
            "symbol": symbol,
            "quantity": quantity,
            "status": "EXECUTED",
            "execution_price": random.uniform(1.0, 100.0),
            "timestamp": time.time()
        }
    
    # Execute the trade with circuit breaker protection
    return await cb.execute(execute_trade)


# Example 3: Using timeout_handler decorator
@timeout_handler(timeout_seconds=2.0, operation_name="process_order")
async def process_order(order_id: str) -> Dict[str, Any]:
    """
    Example function that processes an order with timeout protection.
    
    Args:
        order_id: The order ID
        
    Returns:
        Order processing result
    """
    logger.info(f"Processing order {order_id}")
    
    # Simulate variable processing time that might exceed timeout
    sleep_time = random.uniform(0.1, 3.0)
    await asyncio.sleep(sleep_time)
    
    return {
        "order_id": order_id,
        "status": "PROCESSED",
        "processing_time": sleep_time,
        "timestamp": time.time()
    }


# Example 4: Using bulkhead decorator
@bulkhead(name="risk-engine", max_concurrent=3, max_waiting=5, wait_timeout=1.0)
async def calculate_risk_exposure(portfolio_id: str) -> Dict[str, float]:
    """
    Example function that calculates risk exposure with bulkhead protection.
    
    Args:
        portfolio_id: The portfolio ID
        
    Returns:
        Dictionary of risk metrics
    """
    logger.info(f"Calculating risk exposure for portfolio {portfolio_id}")
    
    # Simulate intensive computation
    await asyncio.sleep(random.uniform(0.5, 2.0))
    
    return {
        "portfolio_id": portfolio_id,
        "var_95": random.uniform(0.01, 0.05),
        "var_99": random.uniform(0.05, 0.1),
        "expected_shortfall": random.uniform(0.1, 0.2),
        "timestamp": time.time()
    }


# Example 5: Combining multiple resilience patterns
@retry_with_policy(max_attempts=2, base_delay=0.5)
@timeout_handler(timeout_seconds=3.0)
@bulkhead(name="api-calls", max_concurrent=5)
async def fetch_external_api_data(api_endpoint: str) -> Dict[str, Any]:
    """
    Example function that fetches data from external API with multiple resilience patterns.
    
    This demonstrates how to combine retry policy, timeout handler, and bulkhead patterns.
    
    Args:
        api_endpoint: The API endpoint to call
        
    Returns:
        API response data
    """
    logger.info(f"Fetching data from external API: {api_endpoint}")
    
    # Simulate API call
    if random.random() < 0.3:  # 30% chance of failure
        logger.error(f"Simulated API call failure: {api_endpoint}")
        raise ConnectionError("Simulated API connection error")
        
    # Simulate variable response time
    await asyncio.sleep(random.uniform(0.1, 2.5))
    
    return {
        "api_endpoint": api_endpoint,
        "data": {
            "items": random.randint(1, 100),
            "status": "success"
        },
        "timestamp": time.time()
    }


# Individual example runners to reduce cognitive complexity
async def run_retry_example() -> None:
    """Run retry policy example."""
    try:
        logger.info("EXAMPLE 1: Retry Policy")
        market_data = await fetch_market_data("EUR/USD")
        logger.info(f"Market data fetched successfully: {market_data}")
    except Exception as e:
        logger.error(f"Failed to fetch market data after retries: {e}")


async def run_circuit_breaker_example() -> None:
    """Run circuit breaker example."""
    try:
        logger.info("\nEXAMPLE 2: Circuit Breaker")
        # Try multiple executions to trigger the circuit breaker
        for i in range(5):
            try:
                trade_result = await execute_trade_with_circuit_breaker(
                    f"ORD-{i+1}", "EUR/USD", random.uniform(1000, 10000)
                )
                logger.info(f"Trade executed successfully: {trade_result}")
            except Exception as e:
                logger.error(f"Trade execution failed: {e}")
    except Exception as e:
        logger.error(f"Circuit breaker example failed: {e}")


async def run_timeout_example() -> None:
    """Run timeout handler example."""
    try:
        logger.info("\nEXAMPLE 3: Timeout Handler")
        # Try multiple executions to demonstrate timeout behavior
        for i in range(3):
            try:
                order_result = await process_order(f"ORD-{i+1}")
                logger.info(f"Order processed within timeout: {order_result}")
            except Exception as e:
                logger.error(f"Order processing timed out: {e}")
    except Exception as e:
        logger.error(f"Timeout handler example failed: {e}")


async def run_bulkhead_example() -> None:
    """Run bulkhead pattern example."""
    try:
        logger.info("\nEXAMPLE 4: Bulkhead Pattern")
        # Launch multiple concurrent executions to demonstrate bulkhead
        tasks = [
            calculate_risk_exposure(f"PORT-{i+1}") 
            for i in range(10)  # Try 10 concurrent calculations
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Risk calculation {i+1} failed: {result}")
            else:
                logger.info(f"Risk calculation {i+1} succeeded: {result}")
    except Exception as e:
        logger.error(f"Bulkhead example failed: {e}")


async def run_combined_patterns_example() -> None:
    """Run combined resilience patterns example."""
    try:
        logger.info("\nEXAMPLE 5: Combined Resilience Patterns")
        endpoints = [f"api/resource/{i}" for i in range(1, 6)]
        
        for endpoint in endpoints:
            try:
                api_data = await fetch_external_api_data(endpoint)
                logger.info(f"API data fetched successfully: {api_data}")
            except Exception as e:
                logger.error(f"API data fetch failed: {e}")
    except Exception as e:
        logger.error(f"Combined patterns example failed: {e}")


# Main example runner
async def run_examples() -> None:
    """Run all examples to demonstrate resilience patterns."""
    # Run each example individually to reduce function complexity
    await run_retry_example()
    await run_circuit_breaker_example()
    await run_timeout_example()
    await run_bulkhead_example()
    await run_combined_patterns_example()


# Main entry point
if __name__ == "__main__":
    asyncio.run(run_examples())
