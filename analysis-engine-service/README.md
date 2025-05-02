# Analysis Engine Service

## Overview
The Analysis Engine Service is a core component of the Forex Trading Platform responsible for performing advanced time-series analysis on market data. This service provides analytical capabilities including pattern recognition, technical indicators, and data transformations needed by other platform services.

## Setup

### Prerequisites
- Python 3.10 or higher
- Poetry (dependency management)
- Access to required data sources
- Network connectivity to other platform services

### Installation
1. Clone the repository
2. Navigate to the service directory:
```bash
cd analysis-engine-service
```
3. Install dependencies using Poetry:
```bash
poetry install
```

### Environment Variables
The following environment variables are required. See `.env.example` for a complete list:

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `PORT` | Service port | 8002 |
| `FEATURE_STORE_BASE_URL` | URL to the Feature Store Service | http://feature-store-service:8000 |
| `DATA_PIPELINE_URL` | URL to the Data Pipeline Service | http://data-pipeline-service:8000 |
| `ML_INTEGRATION_URL` | URL to the ML Integration Service | http://ml-integration-service:8000 |
| `STRATEGY_ENGINE_URL` | URL to the Strategy Execution Engine | http://strategy-execution-engine:8000 |
| `API_KEY` | API key for authentication | - |
| `REDIS_URL` | Redis connection string for caching (optional) | redis://localhost:6379 |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka bootstrap servers (optional) | localhost:9092 |

### Running the Service
Run the service using Poetry:
```bash
poetry run python main.py
```

For development with auto-reload:
```bash
poetry run uvicorn main:app --reload
```

## API Documentation

### Endpoints

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

#### GET /api/v1/analysis/{symbol}
Get analysis for a specific symbol.

**Parameters:**
- `symbol` (path): The trading symbol to analyze
- `timeframe` (query): The timeframe for analysis (e.g., "1h", "1d")
- `indicators` (query): Comma-separated list of indicators to calculate

**Response:**
```json
{
  "symbol": "EUR/USD",
  "timestamp": "2025-04-29T12:00:00Z",
  "data": {
    "indicators": {
      "sma": [1.1012, 1.1015, 1.1020],
      "rsi": [45.2, 46.8, 48.3]
    },
    "patterns": [
      {
        "name": "double_bottom",
        "confidence": 0.87,
        "location": [23, 45]
      }
    ]
  }
}
```

#### POST /api/v1/enhanced-data
Get enhanced data with multiple analysis components.

**Request Body:**
```json
{
  "symbol": "EUR/USD",
  "timeframe": "1h",
  "start_date": "2025-03-01T00:00:00Z",
  "end_date": "2025-04-01T00:00:00Z",
  "indicators": ["sma", "rsi", "macd"],
  "include_patterns": true
}
```

## Integration with Other Services
The Analysis Engine Service integrates with:

- Feature Store Service for retrieving indicator data
- Strategy Execution Engine for providing analytical insights
- ML Integration Service for model-based analytics

## Error Handling
The service implements a standardized exception hierarchy that extends from `common_lib.exceptions`. All errors are returned in a consistent JSON format with appropriate HTTP status codes.

For detailed information on error handling, see [Error Handling Documentation](./docs/error_handling.md).

## Async Patterns
The service uses standardized asynchronous programming patterns throughout the codebase, including async service methods, async analyzer components, and asyncio-based schedulers instead of threading.

For detailed information on async patterns, see [Async Patterns Documentation](./docs/async_patterns.md).

Example error response:
```json
{
  "error_type": "DataFetchError",
  "error_code": "DATA_FETCH_ERROR",
  "message": "Failed to fetch market data from external provider",
  "details": {
    "symbol": "EUR/USD",
    "source": "external_api",
    "status_code": 503
  }
}
```

## Testing

The Analysis Engine Service has comprehensive test coverage including unit tests for core analyzers, integration tests for service interactions, and API endpoint tests.

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage report
poetry run pytest --cov=analysis_engine

# Run specific test categories
poetry run pytest tests/analysis/  # Unit tests for analyzers
poetry run pytest tests/api/       # API endpoint tests
poetry run pytest tests/integration/  # Integration tests
```

For detailed information about the test suite, see [Testing Documentation](./docs/testing.md).

## Security
- Authentication is handled using API keys through `common_lib.security`
- All sensitive configuration is loaded via environment variables
- No hardcoded secrets in the codebase
- Communication between services uses secure channels

## Code Organization

### Configuration Management
The service uses a centralized configuration management system through `analysis_engine.config`. This module provides a comprehensive settings management system with validation and environment variable support.

**Note:** The legacy configuration modules (`analysis_engine.core.config` and `config.config`) are deprecated and will be removed after 2023-12-31. Please use `analysis_engine.config` instead.

For migration guidance, see [Configuration Migration Guide](./docs/configuration_migration_guide.md).

### API Routing
API routes are managed through the `analysis_engine.api.routes` module, which provides a `setup_routes()` function to register all API endpoints with the FastAPI application.

**Note:** The legacy router module (`analysis_engine.api.router`) is deprecated and will be removed after 2023-12-31. Please use `analysis_engine.api.routes` instead.

For migration guidance, see [API Router Migration Guide](./docs/api_router_migration_guide.md).
