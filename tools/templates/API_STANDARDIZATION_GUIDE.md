# API Standardization Guide

This guide provides step-by-step instructions for standardizing API endpoints across the Forex Trading Platform.

## Overview

The platform's API design standards require all endpoints to follow a consistent URL structure, naming conventions, and implementation patterns. This guide will help you migrate existing endpoints to comply with these standards.

## Step 1: Understand the Standards

Before starting, make sure you understand the [API Design Standards](./API_DESIGN_STANDARDS.md).

Key points:
- URL structure: `/v{version}/{service}/{resource}/{id?}/{sub-resource?}`
- Resource names should be plural and use kebab-case
- HTTP methods should be used according to their semantic meaning
- Actions should use POST method and follow the pattern `/v{version}/{service}/{resource}/{id}/{action}`

## Step 2: Analyze Current Endpoints

Use the validation script to identify non-compliant endpoints:

```bash
python tools/linting/validate_api_endpoints.py --service <service-name> --report
```

This will generate a report of all non-compliant endpoints and the specific violations.

## Step 3: Create Standardized Endpoints

For each non-compliant endpoint, create a new standardized version:

### FastAPI Example

Original endpoint:
```python
@router.get("/health", response_model=ServiceHealth)
async def health_check(
    health_check: HealthCheck = Depends(get_health_check)
) -> ServiceHealth:
    """Get detailed health status of the service."""
    return await health_check.check_health()
```

Standardized endpoint:
```python
@router.get(
    "/v1/analysis/health-checks",
    response_model=ServiceHealth,
    summary="Get detailed health status",
    description="Get detailed health status of the service including all components and dependencies."
)
async def health_check(
    health_check: HealthCheck = Depends(get_health_check)
) -> ServiceHealth:
    """
    Get detailed health status of the service.
    
    Returns:
        ServiceHealth object containing detailed health information
    
    Raises:
        HTTPException: If there's an error checking health
    """
    try:
        return await health_check.check_health()
    except Exception as e:
        logger.error(f"Error checking health: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Express.js Example

Original endpoint:
```javascript
router.get('/health', (req, res) => {
  res.json({ status: 'healthy' });
});
```

Standardized endpoint:
```javascript
/**
 * @route GET /v1/service-name/health-checks
 * @description Get detailed health status of the service
 * @returns {object} Health status
 */
router.get('/v1/service-name/health-checks', async (req, res, next) => {
  try {
    const healthStatus = await checkHealth();
    res.status(200).json(healthStatus);
  } catch (error) {
    logger.error(`Error checking health: ${error.message}`);
    next(error);
  }
});
```

## Step 4: Add Backward Compatibility

To ensure a smooth transition, maintain backward compatibility with the old endpoints:

### FastAPI Example

```python
@router.get("/health")
async def health_check_legacy():
    """
    Legacy health check endpoint for backward compatibility.
    
    Use /v1/analysis/health-checks instead.
    """
    logger.info("Legacy health check endpoint called - consider migrating to /v1/analysis/health-checks")
    return await health_check()
```

### Express.js Example

```javascript
router.get('/health', (req, res) => {
  logger.info('Legacy health endpoint called - consider migrating to /v1/service-name/health-checks');
  res.redirect('/v1/service-name/health-checks');
});
```

## Step 5: Update Client Code

Gradually update client code to use the new standardized endpoints:

1. Identify all clients that use the old endpoints
2. Update them to use the new endpoints
3. Monitor for errors during the transition period

## Step 6: Remove Legacy Endpoints

Once all clients have been updated, you can remove the legacy endpoints:

1. Add deprecation notices to legacy endpoints
2. Monitor usage to ensure no clients are still using them
3. Remove legacy endpoints after a suitable transition period

## Templates and Examples

For implementation examples, see:

- [FastAPI Endpoint Template](./fastapi_endpoint_template.py)
- [Express Endpoint Template](./express_endpoint_template.js)

## Tools

The following tools can help with the standardization process:

- `tools/linting/validate_api_endpoints.py`: Validates endpoints against the standards
- `tools/fixing/standardize_api_endpoints.py`: Generates a migration plan for non-compliant endpoints
- `tools/fixing/migrate_api_endpoint.py`: Helps migrate individual endpoint files

## Best Practices

- Standardize one service at a time
- Start with the most critical or frequently used endpoints
- Add comprehensive documentation for all new endpoints
- Use the standardized templates for all new endpoints
- Run the validation script regularly to ensure compliance