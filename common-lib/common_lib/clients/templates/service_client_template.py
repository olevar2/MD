"""
Standardized Service Client Template

This module provides a template for creating service clients that follow the
platform's standardized patterns for service communication, error handling,
resilience, and metrics collection.

Usage:
1. Copy this template to your service's clients directory
2. Rename the class to match your service (e.g., MarketDataClient)
3. Implement service-specific methods using the base HTTP methods
4. Create a factory function in your service's client_factory.py

Example:
```python
from common_lib.clients import BaseServiceClient, ClientConfig
from typing import Dict, Any, Optional, Union, List

class ExampleServiceClient(BaseServiceClient):
    \"\"\"Client for interacting with Example Service.\"\"\"
    
    def __init__(self, config: Union[ClientConfig, Dict[str, Any]]):
        \"\"\"Initialize the client.\"\"\"
        super().__init__(config)
    
    async def get_resource(self, resource_id: str) -> Dict[str, Any]:
        \"\"\"Get a resource by ID.\"\"\"
        return await self.get(f"resources/{resource_id}")
```
"""

import logging
from typing import Dict, Any, Optional, Union, List, TypeVar, Generic
from datetime import datetime

from common_lib.clients.base_client import BaseServiceClient, ClientConfig
from common_lib.clients.exceptions import (
    ClientError,
    ClientConnectionError,
    ClientTimeoutError,
    ClientValidationError,
    ClientAuthenticationError
)

# Type variable for response data
T = TypeVar('T')


class StandardServiceClient(BaseServiceClient, Generic[T]):
    """
    Standardized service client template.
    
    This class extends BaseServiceClient with additional standardized methods
    and patterns for service communication.
    
    Features:
    1. Consistent error handling and logging
    2. Type hints for response data
    3. Standardized method signatures
    4. Built-in request/response logging
    5. Correlation ID propagation
    """
    
    def __init__(self, config: Union[ClientConfig, Dict[str, Any]]):
        """
        Initialize the standardized service client.
        
        Args:
            config: Client configuration
        """
        super().__init__(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def get_resource(self, resource_id: str) -> Dict[str, Any]:
        """
        Template method for getting a resource by ID.
        
        Args:
            resource_id: Resource identifier
            
        Returns:
            Resource data
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientValidationError: If the request is invalid
            ClientAuthenticationError: If authentication fails
        """
        self.logger.debug(f"Getting resource {resource_id}")
        try:
            return await self.get(f"resources/{resource_id}")
        except Exception as e:
            self.logger.error(f"Failed to get resource {resource_id}: {str(e)}")
            # Re-raise the exception to be handled by the caller
            raise
    
    async def create_resource(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Template method for creating a resource.
        
        Args:
            data: Resource data
            
        Returns:
            Created resource data
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientValidationError: If the request is invalid
            ClientAuthenticationError: If authentication fails
        """
        self.logger.debug(f"Creating resource")
        try:
            return await self.post("resources", data=data)
        except Exception as e:
            self.logger.error(f"Failed to create resource: {str(e)}")
            # Re-raise the exception to be handled by the caller
            raise
    
    async def update_resource(self, resource_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Template method for updating a resource.
        
        Args:
            resource_id: Resource identifier
            data: Updated resource data
            
        Returns:
            Updated resource data
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientValidationError: If the request is invalid
            ClientAuthenticationError: If authentication fails
        """
        self.logger.debug(f"Updating resource {resource_id}")
        try:
            return await self.put(f"resources/{resource_id}", data=data)
        except Exception as e:
            self.logger.error(f"Failed to update resource {resource_id}: {str(e)}")
            # Re-raise the exception to be handled by the caller
            raise
    
    async def delete_resource(self, resource_id: str) -> Dict[str, Any]:
        """
        Template method for deleting a resource.
        
        Args:
            resource_id: Resource identifier
            
        Returns:
            Deletion confirmation
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientValidationError: If the request is invalid
            ClientAuthenticationError: If authentication fails
        """
        self.logger.debug(f"Deleting resource {resource_id}")
        try:
            return await self.delete(f"resources/{resource_id}")
        except Exception as e:
            self.logger.error(f"Failed to delete resource {resource_id}: {str(e)}")
            # Re-raise the exception to be handled by the caller
            raise
    
    async def list_resources(
        self, 
        filters: Optional[Dict[str, Any]] = None, 
        page: int = 1, 
        page_size: int = 100
    ) -> Dict[str, Any]:
        """
        Template method for listing resources.
        
        Args:
            filters: Optional filters to apply
            page: Page number
            page_size: Number of items per page
            
        Returns:
            List of resources
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientValidationError: If the request is invalid
            ClientAuthenticationError: If authentication fails
        """
        self.logger.debug(f"Listing resources (page={page}, page_size={page_size})")
        params = {"page": page, "page_size": page_size}
        if filters:
            params.update(filters)
            
        try:
            return await self.get("resources", params=params)
        except Exception as e:
            self.logger.error(f"Failed to list resources: {str(e)}")
            # Re-raise the exception to be handled by the caller
            raise
    
    async def execute_operation(
        self, 
        operation: str, 
        resource_id: Optional[str] = None, 
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Template method for executing a custom operation.
        
        Args:
            operation: Operation name
            resource_id: Optional resource identifier
            data: Optional operation data
            
        Returns:
            Operation result
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientValidationError: If the request is invalid
            ClientAuthenticationError: If authentication fails
        """
        endpoint = f"operations/{operation}"
        if resource_id:
            endpoint = f"resources/{resource_id}/{operation}"
            
        self.logger.debug(f"Executing operation {operation}")
        try:
            return await self.post(endpoint, data=data)
        except Exception as e:
            self.logger.error(f"Failed to execute operation {operation}: {str(e)}")
            # Re-raise the exception to be handled by the caller
            raise
    
    async def get_health(self) -> Dict[str, Any]:
        """
        Check the health of the service.
        
        Returns:
            Health status
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
        """
        self.logger.debug("Checking service health")
        try:
            return await self.get("health")
        except Exception as e:
            self.logger.error(f"Failed to check service health: {str(e)}")
            # Re-raise the exception to be handled by the caller
            raise
    
    async def with_correlation_id(self, correlation_id: str) -> 'StandardServiceClient[T]':
        """
        Create a new client instance with the specified correlation ID.
        
        This method allows for easy propagation of correlation IDs across service calls.
        
        Args:
            correlation_id: Correlation ID to use for requests
            
        Returns:
            New client instance with the correlation ID set
        """
        # Create a copy of the configuration
        config_dict = self.config.model_dump()
        
        # Update headers with correlation ID
        headers = config_dict.get("default_headers", {})
        headers["X-Correlation-ID"] = correlation_id
        config_dict["default_headers"] = headers
        
        # Create new client with updated configuration
        return self.__class__(config_dict)