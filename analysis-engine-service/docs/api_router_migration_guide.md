# API Router Migration Guide

This guide provides instructions for migrating from the deprecated `analysis_engine.api.router` module to the recommended `analysis_engine.api.routes` module.

## Background

The Analysis Engine Service previously had two similar modules for API routing:

1. `analysis_engine.api.router` - Creates an `api_router` with all v1 API endpoints
2. `analysis_engine.api.routes` - Provides a `setup_routes()` function that registers all API endpoints with the FastAPI app

To reduce duplication and simplify the codebase, we are consolidating these modules and standardizing on `analysis_engine.api.routes`.

## Migration Steps

### If you're using `api_router` directly

#### Before:

```python
from analysis_engine.api.router import api_router

# In your FastAPI app setup
app = FastAPI()
app.include_router(api_router)
```

#### After:

```python
from analysis_engine.api.routes import setup_routes

# In your FastAPI app setup
app = FastAPI()
setup_routes(app)
```

### If you need direct access to specific routers

#### Before:

```python
from analysis_engine.api.router import api_router
# Or importing specific routers from their modules
from analysis_engine.api.v1.analysis_results_api import router as analysis_results_router
```

#### After:

```python
# Import specific routers directly from their modules
from analysis_engine.api.v1.analysis_results_api import router as analysis_results_router
```

## Benefits of Using `setup_routes()`

1. **Comprehensive Setup**: Includes all API endpoints, including root and health check endpoints
2. **Consistent Prefixing**: Ensures all endpoints have consistent URL prefixes
3. **Centralized Configuration**: All route setup is managed in one place
4. **Future-Proof**: New endpoints will be added to `setup_routes()` first

## Timeline

The `analysis_engine.api.router` module is deprecated and will be removed after December 31, 2023. Please update your code before this date to avoid disruption.

## Need Help?

If you encounter any issues during migration, please contact the Analysis Engine team or open a support ticket.
