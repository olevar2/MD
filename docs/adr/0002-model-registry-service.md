# ADR-0002: Model Registry Service Creation

## Status

Accepted

## Context

Our ML-related services (ml-workbench-service, ml-integration-service, and analysis-engine-service) had several issues:

1. Circular dependencies between services
2. Inconsistent model versioning and lifecycle management
3. Duplicate model metadata storage
4. No centralized model deployment control
5. Complex A/B testing implementation spread across services

## Decision

We created a dedicated Model Registry Service with the following responsibilities:

### 1. Core Domain Models
```python
class ModelType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    REINFORCEMENT = "reinforcement"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"

class ModelStage(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
```

### 2. Architecture
```
model-registry-service/
├── api/                # FastAPI endpoints
├── core/              # Core business logic
├── domain/           # Domain models and interfaces
└── infrastructure/   # Infrastructure implementations
```

### 3. Key Features
- Model versioning and metadata tracking
- Model lifecycle management
- Model artifact storage
- Model A/B testing support
- Model metrics tracking
- Model deployment management

### 4. Integration Points
- ML Workbench: Uses for model training and experimentation
- ML Integration: Uses for model deployment and serving
- Analysis Engine: Uses for retrieving production models
- Strategy Execution: Uses for loading trading strategy models

## Consequences

### Positive
1. Eliminated circular dependencies between ML services
2. Centralized model management
3. Consistent model versioning
4. Simplified A/B testing
5. Clear separation of concerns

### Negative
1. New service to maintain
2. Additional deployment complexity
3. Need for data migration
4. Potential performance overhead
5. Learning curve for teams

## Technical Implementation

### 1. Model Storage
- Uses filesystem-based storage with clear directory structure
- Supports future cloud storage implementations
- Handles model versioning and artifacts

### 2. API Design
- RESTful endpoints for model management
- Websocket notifications for model updates
- Clear versioning strategy
- Comprehensive documentation

### 3. Integration
- Common interface in common-lib
- Adapter implementations in each service
- Event-based notifications
- Circuit breaker pattern

## Migration Plan

1. Create initial service implementation
2. Migrate existing models and metadata
3. Update dependent services
4. Gradually deprecate old implementations
5. Monitor and validate changes

## Validation

Service must demonstrate:
1. Zero circular dependencies
2. Consistent model versioning
3. Reliable A/B testing
4. Clear audit trail
5. Performance within SLAs
