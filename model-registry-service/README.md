# Model Registry Service

A dedicated service for managing machine learning model versioning, deployment, and lifecycle in the Forex Trading Platform.

## Overview

The Model Registry Service provides a centralized repository for ML models with features including:

- Model versioning and metadata tracking
- Model lifecycle management (development, staging, production)
- Model artifact storage and retrieval
- Model A/B testing support
- Model metrics tracking
- Model deployment management

## Architecture

The service follows a clean architecture pattern with clear separation of concerns:

```
model_registry/
├── api/                # API layer (FastAPI endpoints)
├── core/              # Core business logic
├── domain/           # Domain models and interfaces
└── infrastructure/   # Infrastructure implementations
```

## Key Features

### Model Registration
```http
POST /models
{
    "name": "forex_price_predictor",
    "model_type": "forecasting",
    "description": "LSTM model for forex price prediction",
    "business_domain": "forex_trading",
    "purpose": "price_prediction"
}
```

### Version Creation
```http
POST /models/{model_id}/versions
{
    "framework": "pytorch",
    "framework_version": "2.0.0",
    "description": "Improved LSTM architecture",
    "metrics": {
        "mae": 0.0012,
        "rmse": 0.0015
    }
}
```

### Stage Updates
```http
PATCH /versions/{version_id}/stage
{
    "stage": "production"
}
```

## Integration

The Model Registry Service eliminates circular dependencies by providing a central service that other services can depend on:

- ml-workbench-service: Uses for model training and experimentation
- ml-integration-service: Uses for model deployment and serving
- analysis-engine-service: Uses for retrieving production models
- strategy-execution-engine: Uses for loading trading strategy models

## Setup

1. Install dependencies:
```powershell
poetry install
```

2. Configure environment variables:
```powershell
$env:MODEL_REGISTRY_STORAGE_PATH = "D:/path/to/storage"
```

3. Run the service:
```powershell
poetry run uvicorn model_registry.api.main:app --host 0.0.0.0 --port 8000
```

## API Documentation

When running, API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Error Handling

All errors follow a standardized format:
```json
{
    "detail": "Error message describing what went wrong"
}
```

HTTP status codes:
- 400: Invalid request (e.g., invalid model data)
- 404: Resource not found
- 500: Internal server error