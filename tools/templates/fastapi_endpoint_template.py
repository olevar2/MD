"""
Standardized FastAPI Endpoint Template

This template demonstrates how to create API endpoints that follow
the platform's standardized API design patterns.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from common_lib.logging import get_logger
from common_lib.errors import ServiceError

# Initialize logger
logger = get_logger(__name__)

# Define response models
class ResourceResponse(BaseModel):
    """Response model for a single resource."""
    id: str = Field(..., description="Unique identifier for the resource")
    name: str = Field(..., description="Name of the resource")
    description: Optional[str] = Field(None, description="Description of the resource")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "resource-123",
                "name": "Example Resource",
                "description": "This is an example resource",
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-02T00:00:00Z"
            }
        }

class ResourceListResponse(BaseModel):
    """Response model for a list of resources."""
    items: List[ResourceResponse] = Field(..., description="List of resources")
    total: int = Field(..., description="Total number of resources")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    
    class Config:
        schema_extra = {
            "example": {
                "items": [
                    {
                        "id": "resource-123",
                        "name": "Example Resource 1",
                        "description": "This is an example resource",
                        "created_at": "2023-01-01T00:00:00Z",
                        "updated_at": "2023-01-02T00:00:00Z"
                    },
                    {
                        "id": "resource-456",
                        "name": "Example Resource 2",
                        "description": "This is another example resource",
                        "created_at": "2023-01-03T00:00:00Z",
                        "updated_at": "2023-01-04T00:00:00Z"
                    }
                ],
                "total": 2,
                "page": 1,
                "page_size": 10
            }
        }

class CreateResourceRequest(BaseModel):
    """Request model for creating a resource."""
    name: str = Field(..., description="Name of the resource")
    description: Optional[str] = Field(None, description="Description of the resource")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "New Resource",
                "description": "This is a new resource"
            }
        }

class UpdateResourceRequest(BaseModel):
    """Request model for updating a resource."""
    name: Optional[str] = Field(None, description="Name of the resource")
    description: Optional[str] = Field(None, description="Description of the resource")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Updated Resource",
                "description": "This is an updated resource"
            }
        }

# Create router with standardized prefix
# Format: /v{version}/{service}/{resource}
router = APIRouter(
    prefix="/v1/service-name/resources",
    tags=["Resources"]
)

@router.get(
    "",
    response_model=ResourceListResponse,
    summary="List resources",
    description="Get a paginated list of resources with optional filtering."
)
async def list_resources(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    name: Optional[str] = Query(None, description="Filter by name")
) -> ResourceListResponse:
    """
    Get a paginated list of resources with optional filtering.
    
    Args:
        page: Page number (starts at 1)
        page_size: Number of items per page (max 100)
        name: Optional filter by name
        
    Returns:
        ResourceListResponse: Paginated list of resources
        
    Raises:
        HTTPException: If there's an error retrieving resources
    """
    try:
        # Implementation logic here
        # This is just an example
        items = [
            ResourceResponse(
                id=f"resource-{i}",
                name=f"Example Resource {i}",
                description=f"This is example resource {i}",
                created_at="2023-01-01T00:00:00Z",
                updated_at="2023-01-02T00:00:00Z"
            )
            for i in range(1, 3)
        ]
        
        return ResourceListResponse(
            items=items,
            total=len(items),
            page=page,
            page_size=page_size
        )
    except Exception as e:
        logger.error(f"Error listing resources: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing resources: {str(e)}")

@router.get(
    "/{resource_id}",
    response_model=ResourceResponse,
    summary="Get resource",
    description="Get a specific resource by ID."
)
async def get_resource(
    resource_id: str = Path(..., description="Resource ID")
) -> ResourceResponse:
    """
    Get a specific resource by ID.
    
    Args:
        resource_id: Resource ID
        
    Returns:
        ResourceResponse: Resource details
        
    Raises:
        HTTPException: If resource not found or there's an error retrieving it
    """
    try:
        # Implementation logic here
        # This is just an example
        return ResourceResponse(
            id=resource_id,
            name="Example Resource",
            description="This is an example resource",
            created_at="2023-01-01T00:00:00Z",
            updated_at="2023-01-02T00:00:00Z"
        )
    except Exception as e:
        logger.error(f"Error getting resource {resource_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Resource {resource_id} not found")

@router.post(
    "",
    response_model=ResourceResponse,
    status_code=201,
    summary="Create resource",
    description="Create a new resource."
)
async def create_resource(
    request: CreateResourceRequest
) -> ResourceResponse:
    """
    Create a new resource.
    
    Args:
        request: Resource creation request
        
    Returns:
        ResourceResponse: Created resource
        
    Raises:
        HTTPException: If there's an error creating the resource
    """
    try:
        # Implementation logic here
        # This is just an example
        return ResourceResponse(
            id="resource-new",
            name=request.name,
            description=request.description,
            created_at="2023-01-01T00:00:00Z",
            updated_at="2023-01-01T00:00:00Z"
        )
    except Exception as e:
        logger.error(f"Error creating resource: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating resource: {str(e)}")

@router.put(
    "/{resource_id}",
    response_model=ResourceResponse,
    summary="Update resource",
    description="Update an existing resource."
)
async def update_resource(
    resource_id: str = Path(..., description="Resource ID"),
    request: UpdateResourceRequest = None
) -> ResourceResponse:
    """
    Update an existing resource.
    
    Args:
        resource_id: Resource ID
        request: Resource update request
        
    Returns:
        ResourceResponse: Updated resource
        
    Raises:
        HTTPException: If resource not found or there's an error updating it
    """
    try:
        # Implementation logic here
        # This is just an example
        return ResourceResponse(
            id=resource_id,
            name=request.name or "Example Resource",
            description=request.description or "This is an updated resource",
            created_at="2023-01-01T00:00:00Z",
            updated_at="2023-01-02T00:00:00Z"
        )
    except Exception as e:
        logger.error(f"Error updating resource {resource_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Resource {resource_id} not found")

@router.delete(
    "/{resource_id}",
    status_code=204,
    summary="Delete resource",
    description="Delete an existing resource."
)
async def delete_resource(
    resource_id: str = Path(..., description="Resource ID")
) -> None:
    """
    Delete an existing resource.
    
    Args:
        resource_id: Resource ID
        
    Returns:
        None
        
    Raises:
        HTTPException: If resource not found or there's an error deleting it
    """
    try:
        # Implementation logic here
        # This is just an example
        return None
    except Exception as e:
        logger.error(f"Error deleting resource {resource_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Resource {resource_id} not found")

@router.post(
    "/{resource_id}/actions/validate",
    response_model=Dict[str, Any],
    summary="Validate resource",
    description="Perform validation on a resource."
)
async def validate_resource(
    resource_id: str = Path(..., description="Resource ID")
) -> Dict[str, Any]:
    """
    Perform validation on a resource.
    
    This is an example of an action endpoint. Actions should:
    1. Use POST method
    2. Follow the pattern /v{version}/{service}/{resource}/{id}/{action}
    3. Return appropriate response models
    
    Args:
        resource_id: Resource ID
        
    Returns:
        Dict[str, Any]: Validation results
        
    Raises:
        HTTPException: If resource not found or there's an error validating it
    """
    try:
        # Implementation logic here
        # This is just an example
        return {
            "resource_id": resource_id,
            "is_valid": True,
            "validation_timestamp": "2023-01-01T00:00:00Z",
            "validation_results": [
                {"check": "format", "passed": True},
                {"check": "content", "passed": True}
            ]
        }
    except Exception as e:
        logger.error(f"Error validating resource {resource_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Resource {resource_id} not found")