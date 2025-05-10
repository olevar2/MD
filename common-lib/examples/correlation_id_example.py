"""
Example usage of correlation ID utility.

This example demonstrates how to use the correlation ID utility in different scenarios:
1. HTTP requests with FastAPI
2. Service-to-service communication
3. Event-based communication
4. Logging with correlation IDs
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware

from common_lib.correlation import (
    FastAPICorrelationIdMiddleware,
    get_correlation_id,
    set_correlation_id,
    correlation_id_context,
    async_correlation_id_context,
    with_correlation_id,
    with_async_correlation_id,
    add_correlation_to_event_metadata,
    extract_correlation_id_from_event,
    with_event_correlation,
    with_async_event_correlation,
    CORRELATION_ID_HEADER
)
from common_lib.clients import BaseServiceClient, ClientConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create a logger that includes correlation ID
logger = logging.getLogger("correlation_example")


class CorrelationFilter(logging.Filter):
    """Logging filter that adds correlation ID to log records."""
    
    def filter(self, record):
        correlation_id = get_correlation_id()
        record.correlation_id = correlation_id or "no-correlation-id"
        return True


# Add the correlation filter to the logger
logger.addFilter(CorrelationFilter())


# Create a FastAPI app with correlation ID middleware
app = FastAPI(title="Correlation ID Example")
app.add_middleware(FastAPICorrelationIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Example service client
class ExampleServiceClient(BaseServiceClient):
    """Example service client with correlation ID support."""
    
    def __init__(self, config):
        super().__init__(config)
    
    async def get_data(self, resource_id: str) -> Dict[str, Any]:
        """Get data from the service."""
        logger.info(f"Getting data for resource {resource_id}")
        # The BaseServiceClient automatically adds correlation ID to headers
        return await self.get(f"resources/{resource_id}")


# Create a client instance
example_client = ExampleServiceClient(
    ClientConfig(
        base_url="https://example.com/api",
        service_name="example-service"
    )
)


# Example event handler with correlation ID support
@with_event_correlation
def handle_event(event: Dict[str, Any]) -> None:
    """Handle an event with correlation ID support."""
    # The decorator extracts correlation ID from the event and sets it in the context
    correlation_id = get_correlation_id()
    logger.info(f"Handling event with correlation ID: {correlation_id}")
    
    # Process the event
    event_type = event.get("event_type", "unknown")
    logger.info(f"Processing event of type: {event_type}")


# Example async event handler with correlation ID support
@with_async_event_correlation
async def handle_event_async(event: Dict[str, Any]) -> None:
    """Handle an event asynchronously with correlation ID support."""
    # The decorator extracts correlation ID from the event and sets it in the context
    correlation_id = get_correlation_id()
    logger.info(f"Handling async event with correlation ID: {correlation_id}")
    
    # Process the event
    event_type = event.get("event_type", "unknown")
    logger.info(f"Processing async event of type: {event_type}")


# Example function with correlation ID support
@with_correlation_id
def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process data with correlation ID support."""
    # The decorator ensures a correlation ID is available
    correlation_id = get_correlation_id()
    logger.info(f"Processing data with correlation ID: {correlation_id}")
    
    # Process the data
    result = {"processed": True, "original": data}
    logger.info(f"Data processing complete")
    
    return result


# Example async function with correlation ID support
@with_async_correlation_id
async def process_data_async(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process data asynchronously with correlation ID support."""
    # The decorator ensures a correlation ID is available
    correlation_id = get_correlation_id()
    logger.info(f"Processing data asynchronously with correlation ID: {correlation_id}")
    
    # Simulate async processing
    await asyncio.sleep(0.1)
    
    # Process the data
    result = {"processed": True, "original": data}
    logger.info(f"Async data processing complete")
    
    return result


# FastAPI endpoint that uses correlation ID
@app.get("/api/resources/{resource_id}")
async def get_resource(resource_id: str, request: Request):
    """Get a resource by ID."""
    # The correlation ID is automatically set by the middleware
    correlation_id = get_correlation_id()
    logger.info(f"Received request for resource {resource_id}")
    
    # Call another service with the same correlation ID
    client_with_correlation = example_client.with_correlation_id(correlation_id)
    
    try:
        # Simulate a service call
        # In a real scenario, this would call the actual service
        result = {
            "id": resource_id,
            "name": f"Resource {resource_id}",
            "correlation_id": correlation_id
        }
        
        logger.info(f"Successfully retrieved resource {resource_id}")
        return result
    except Exception as e:
        logger.error(f"Error retrieving resource {resource_id}: {str(e)}")
        raise


# Example of using correlation ID context managers
async def example_context_usage():
    """Example of using correlation ID context managers."""
    # Use the synchronous context manager
    with correlation_id_context("sync-context-id"):
        logger.info("Inside synchronous correlation ID context")
        process_data({"key": "value"})
    
    # Use the asynchronous context manager
    async with async_correlation_id_context("async-context-id"):
        logger.info("Inside asynchronous correlation ID context")
        await process_data_async({"key": "value"})


# Example of creating and handling events with correlation IDs
def example_event_handling():
    """Example of creating and handling events with correlation IDs."""
    # Create an event with correlation ID
    correlation_id = str(uuid.uuid4())
    metadata = add_correlation_to_event_metadata({}, correlation_id)
    
    event = {
        "event_type": "example_event",
        "data": {"key": "value"},
        "metadata": metadata
    }
    
    # Handle the event
    handle_event(event)


# Main function to run the examples
async def run_examples():
    """Run all examples."""
    logger.info("Starting correlation ID examples")
    
    # Example of using context managers
    await example_context_usage()
    
    # Example of event handling
    example_event_handling()
    
    logger.info("Correlation ID examples complete")


# Run the examples when the script is executed directly
if __name__ == "__main__":
    asyncio.run(run_examples())
"""
