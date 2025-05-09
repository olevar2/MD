# API Design Standards

This document outlines the standardized API design patterns for the Forex Trading Platform.

## URL Structure

All API endpoints should follow this URL structure:

```
/v{version}/{service}/{resource}/{id?}/{sub-resource?}
```

Where:
- `{version}`: API version (e.g., `v1`)
- `{service}`: Service name in kebab-case (e.g., `analysis-engine`)
- `{resource}`: Resource name in plural, kebab-case (e.g., `market-regimes`)
- `{id}`: Optional resource identifier
- `{sub-resource}`: Optional sub-resource name in plural, kebab-case

Examples:
- `/v1/analysis-engine/analyzers`
- `/v1/analysis-engine/analyzers/gann-tools`
- `/v1/analysis-engine/analyzers/gann-tools/configurations`

## HTTP Methods

Use standard HTTP methods according to their semantic meaning:

- `GET`: Read operations
- `POST`: Create operations
- `PUT`: Full update operations
- `PATCH`: Partial update operations
- `DELETE`: Delete operations

## Resource Naming

- Use plural nouns for resource names (e.g., `analyzers`, not `analyzer`)
- Use kebab-case for multi-word resource names (e.g., `market-regimes`, not `marketRegimes` or `market_regimes`)
- Be consistent with naming across all services

## Request and Response Formats

### Request Format

- Use JSON for request bodies
- Use query parameters for filtering, pagination, and sorting
- Use path parameters for resource identifiers

Example:
```json
{
  "name": "Example Resource",
  "description": "This is an example resource"
}
```

### Response Format

- Use JSON for response bodies
- Include appropriate HTTP status codes
- Include pagination metadata for list endpoints
- Include error details for error responses

Success Response Example:
```json
{
  "id": "resource-123",
  "name": "Example Resource",
  "description": "This is an example resource",
  "created_at": "2023-01-01T00:00:00Z",
  "updated_at": "2023-01-02T00:00:00Z"
}
```

Error Response Example:
```json
{
  "error": {
    "message": "Resource not found",
    "type": "ResourceNotFoundError",
    "code": "RESOURCE_NOT_FOUND"
  },
  "success": false
}
```

## Pagination

List endpoints should support pagination with the following query parameters:

- `page`: Page number (starts at 1)
- `page_size`: Number of items per page (default: 10, max: 100)

Response should include pagination metadata:

```json
{
  "items": [...],
  "total": 100,
  "page": 1,
  "page_size": 10
}
```

## Filtering and Sorting

- Use query parameters for filtering (e.g., `?name=example`)
- Use `sort` query parameter for sorting (e.g., `?sort=name` or `?sort=-name` for descending)

## Versioning

- Include API version in the URL path (e.g., `/v1/...`)
- Major version changes should be reflected in the URL
- Minor version changes can be handled with headers or query parameters

## Error Handling

- Use appropriate HTTP status codes
- Include detailed error messages
- Include error type and code for programmatic handling
- Log errors with correlation IDs

## Actions

For operations that don't fit the standard CRUD model, use action endpoints:

- Use POST method
- Follow the pattern: `/v{version}/{service}/{resource}/{id}/{action}`
- Use verbs for action names (e.g., `validate`, `analyze`, `process`)

Example: `/v1/analysis-engine/analyzers/gann-tools/validate`

## Documentation

- Include OpenAPI/Swagger documentation for all endpoints
- Document request and response schemas
- Include examples for requests and responses
- Document error responses

## Implementation Templates

For implementation examples, see:

- [FastAPI Endpoint Template](./fastapi_endpoint_template.py)
- [Express Endpoint Template](./express_endpoint_template.js)

## Migration Strategy

When migrating existing endpoints to follow these standards:

1. Create new endpoints following the standards
2. Maintain backward compatibility with old endpoints
3. Add deprecation notices to old endpoints
4. Monitor usage and gradually phase out old endpoints

## Compliance Validation

Use the API endpoint validation script to check compliance with these standards:

```bash
python tools/linting/validate_api_endpoints.py --service <service-name> --report
```