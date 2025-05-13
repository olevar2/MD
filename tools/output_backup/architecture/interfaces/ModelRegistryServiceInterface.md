# ModelRegistryServiceInterface

*Generated on 2025-05-13 05:58:22*

## Description

Interface for model-registry-service service.

## File

`model_registry_service_interface.py`

## Methods

### get_status() -> Dict

Get the status of the service.

Returns:
    Service status information

#### Returns

- Dict

### get_model(model_id: str) -> Dict

Get a model from the registry.

Args:
    model_id: Model identifier
Returns:
    Model information

#### Parameters

- **model_id** (str)

#### Returns

- Dict

### list_models(tags: Optional, limit: int, offset: int) -> Dict

List available models.

Args:
    tags: Filter by tags
Args:
    limit: Maximum number of results
Args:
    offset: Result offset
Returns:
    Dictionary with models and pagination information

#### Parameters

- **tags** (Optional)
- **limit** (int)
- **offset** (int)

#### Returns

- Dict

