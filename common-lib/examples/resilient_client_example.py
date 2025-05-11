"""
Resilient Service Client Example

This example demonstrates how to use the resilient service client in the common-lib package.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List

from common_lib.service_client import (
    ResilientServiceClientConfig,
    ResilientServiceClient
)
from common_lib.errors.base_exceptions import ServiceError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("resilient-client-example")


# Example service client
class ExampleServiceClient:
    """Example service client that uses the resilient service client."""
    
    def __init__(self, service_url: str = "http://example-service:8000"):
        """
        Initialize the service client.
        
        Args:
            service_url: URL of the service
        """
        self.logger = logger.getChild("ExampleServiceClient")
        
        # Create resilient service client
        config = ResilientServiceClientConfig(
            service_name="example-service",
            base_url=service_url,
            timeout=5.0,
            retry_count=3,
            retry_backoff=0.5,
            max_concurrent_requests=10,
            circuit_breaker_threshold=5,
            circuit_breaker_recovery_time=30.0,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )
        self.client = ResilientServiceClient(config, logger=self.logger)
    
    async def connect(self):
        """Connect to the service."""
        await self.client.connect()
    
    async def close(self):
        """Close the connection to the service."""
        await self.client.close()
    
    async def get_item(self, item_id: str) -> Dict[str, Any]:
        """
        Get an item from the service.
        
        Args:
            item_id: ID of the item to get
            
        Returns:
            Item data
            
        Raises:
            ServiceError: If the service call fails
        """
        try:
            return await self.client.get(f"/items/{item_id}")
        except ServiceError as e:
            self.logger.error(f"Failed to get item {item_id}: {str(e)}")
            raise
    
    async def create_item(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an item in the service.
        
        Args:
            item_data: Data for the item to create
            
        Returns:
            Created item data
            
        Raises:
            ServiceError: If the service call fails
        """
        try:
            return await self.client.post("/items", json=item_data)
        except ServiceError as e:
            self.logger.error(f"Failed to create item: {str(e)}")
            raise
    
    async def update_item(self, item_id: str, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an item in the service.
        
        Args:
            item_id: ID of the item to update
            item_data: Data for the item to update
            
        Returns:
            Updated item data
            
        Raises:
            ServiceError: If the service call fails
        """
        try:
            return await self.client.put(f"/items/{item_id}", json=item_data)
        except ServiceError as e:
            self.logger.error(f"Failed to update item {item_id}: {str(e)}")
            raise
    
    async def delete_item(self, item_id: str) -> Dict[str, Any]:
        """
        Delete an item from the service.
        
        Args:
            item_id: ID of the item to delete
            
        Returns:
            Deletion result
            
        Raises:
            ServiceError: If the service call fails
        """
        try:
            return await self.client.delete(f"/items/{item_id}")
        except ServiceError as e:
            self.logger.error(f"Failed to delete item {item_id}: {str(e)}")
            raise
    
    async def search_items(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for items in the service.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of items matching the query
            
        Raises:
            ServiceError: If the service call fails
        """
        try:
            params = {"q": query, "limit": limit}
            result = await self.client.get("/items/search", params=params)
            return result.get("items", [])
        except ServiceError as e:
            self.logger.error(f"Failed to search items with query '{query}': {str(e)}")
            raise


# Mock server for testing
class MockServer:
    """Mock server for testing the resilient service client."""
    
    def __init__(self):
        """Initialize the mock server."""
        self.items = {}
        self.failure_count = 0
        self.max_failures = 3
    
    async def handle_request(self, method: str, path: str, params: Dict[str, Any] = None, json_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle a request to the mock server.
        
        Args:
            method: HTTP method
            path: Request path
            params: Query parameters
            json_data: JSON body
            
        Returns:
            Response data
            
        Raises:
            ServiceError: If the request fails
        """
        # Simulate random failures
        if self.failure_count < self.max_failures:
            self.failure_count += 1
            raise ServiceError("Service temporarily unavailable", error_code=503)
        
        # Reset failure count
        self.failure_count = 0
        
        # Handle request
        if path.startswith("/items/search"):
            # Search items
            query = params.get("q", "")
            limit = int(params.get("limit", 10))
            items = [item for item in self.items.values() if query.lower() in item.get("name", "").lower()]
            return {"items": items[:limit]}
        elif path.startswith("/items/") and method == "GET":
            # Get item
            item_id = path.split("/")[-1]
            if item_id in self.items:
                return self.items[item_id]
            else:
                raise ServiceError(f"Item {item_id} not found", error_code=404)
        elif path.startswith("/items/") and method == "PUT":
            # Update item
            item_id = path.split("/")[-1]
            if item_id in self.items:
                self.items[item_id].update(json_data)
                return self.items[item_id]
            else:
                raise ServiceError(f"Item {item_id} not found", error_code=404)
        elif path.startswith("/items/") and method == "DELETE":
            # Delete item
            item_id = path.split("/")[-1]
            if item_id in self.items:
                item = self.items.pop(item_id)
                return {"success": True, "deleted": item}
            else:
                raise ServiceError(f"Item {item_id} not found", error_code=404)
        elif path == "/items" and method == "POST":
            # Create item
            item_id = json_data.get("id", str(len(self.items) + 1))
            self.items[item_id] = {"id": item_id, **json_data}
            return self.items[item_id]
        else:
            # Unknown path
            raise ServiceError(f"Unknown path: {path}", error_code=404)


# Example usage
async def main():
    """Run the example."""
    # Create mock server
    mock_server = MockServer()
    
    # Create service client
    client = ExampleServiceClient()
    
    # Connect to the service
    await client.connect()
    
    try:
        # Create an item
        logger.info("Creating an item...")
        try:
            item = await client.create_item({"name": "Test Item", "value": 42})
            logger.info(f"Created item: {item}")
        except ServiceError as e:
            logger.error(f"Error creating item: {str(e)}")
        
        # Get the item
        logger.info("\nGetting the item...")
        try:
            item = await client.get_item("1")
            logger.info(f"Got item: {item}")
        except ServiceError as e:
            logger.error(f"Error getting item: {str(e)}")
        
        # Update the item
        logger.info("\nUpdating the item...")
        try:
            item = await client.update_item("1", {"value": 43})
            logger.info(f"Updated item: {item}")
        except ServiceError as e:
            logger.error(f"Error updating item: {str(e)}")
        
        # Search for items
        logger.info("\nSearching for items...")
        try:
            items = await client.search_items("Test")
            logger.info(f"Found items: {items}")
        except ServiceError as e:
            logger.error(f"Error searching items: {str(e)}")
        
        # Delete the item
        logger.info("\nDeleting the item...")
        try:
            result = await client.delete_item("1")
            logger.info(f"Deleted item: {result}")
        except ServiceError as e:
            logger.error(f"Error deleting item: {str(e)}")
    finally:
        # Close the connection
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
