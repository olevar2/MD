"""
Service Client Usage Example

This module demonstrates how to use the standardized service clients.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from examples.service_client_examples.client_factory import (
    initialize_clients,
    get_market_data_client
)
from common_lib.clients.exceptions import (
    ClientError,
    ClientConnectionError,
    ClientTimeoutError,
    ClientValidationError,
    ClientAuthenticationError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("client_example")


async def market_data_example():
    """Example of using the Market Data client."""
    logger.info("Running Market Data client example...")
    
    # Get the client
    client = get_market_data_client()
    
    # Example 1: Get OHLCV data
    try:
        result = await client.get_ohlcv_data(
            symbol="EUR/USD",
            timeframe="1h",
            start_time=datetime.now() - timedelta(days=7),
            end_time=datetime.now(),
            limit=100
        )
        
        logger.info(f"Got {len(result.get('data', []))} OHLCV data points")
    except ClientTimeoutError as e:
        logger.error(f"Request timed out: {str(e)}")
    except ClientConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
    except ClientError as e:
        logger.error(f"Client error: {str(e)}")
    
    # Example 2: Get instrument information
    try:
        instrument = await client.get_instrument("EUR/USD")
        logger.info(f"Got instrument information: {instrument}")
    except ClientError as e:
        logger.error(f"Failed to get instrument information: {str(e)}")
    
    # Example 3: List instruments
    try:
        instruments = await client.list_instruments(category="forex")
        logger.info(f"Got {len(instruments.get('data', []))} forex instruments")
    except ClientError as e:
        logger.error(f"Failed to list instruments: {str(e)}")
    
    # Example 4: Get latest price
    try:
        price = await client.get_latest_price("EUR/USD")
        logger.info(f"Latest EUR/USD price: {price}")
    except ClientError as e:
        logger.error(f"Failed to get latest price: {str(e)}")
    
    # Example 5: Using correlation ID
    try:
        # Create a client with correlation ID
        correlation_id = "example-correlation-id"
        client_with_correlation_id = await client.with_correlation_id(correlation_id)
        
        # Make a request with the correlation ID
        result = await client_with_correlation_id.get_ohlcv_data(
            symbol="EUR/USD",
            timeframe="1h",
            limit=10
        )
        
        logger.info(f"Got {len(result.get('data', []))} OHLCV data points with correlation ID")
    except ClientError as e:
        logger.error(f"Failed to get OHLCV data with correlation ID: {str(e)}")


async def main():
    """Main function."""
    logger.info("Starting service client example...")
    
    # Initialize clients
    initialize_clients()
    
    # Run examples
    await market_data_example()
    
    logger.info("Service client example completed")


if __name__ == "__main__":
    asyncio.run(main())