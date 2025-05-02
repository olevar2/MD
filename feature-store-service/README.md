# Feature Store Service

## Overview
The Feature Store Service is a centralized repository for storing, managing, and serving features used in the Forex Trading Platform. It acts as the canonical source for all indicator implementations, providing consistent data access patterns for machine learning models and analysis components.

## Setup

### Prerequisites
- Python 3.10 or higher
- Poetry (dependency management)
- Redis (for feature caching)
- Database access (PostgreSQL recommended)

### Installation
1. Clone the repository
2. Navigate to the service directory:
```bash
cd feature-store-service
```
3. Install dependencies using Poetry:
```bash
poetry install
```

### Environment Variables
The following environment variables are required:

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `PORT` | Service port | 8001 |
| `DB_CONNECTION_STRING` | Database connection string | postgresql://user:password@localhost:5432/features |
| `REDIS_URL` | Redis connection string for caching | redis://localhost:6379 |
| `API_KEY` | API key for authentication | - |
| `DATA_RETENTION_DAYS` | Days to keep feature data | 90 |

### Running the Service
Run the service using Poetry:
```bash
poetry run python -m feature_store_service.main
```

For development with auto-reload:
```bash
poetry run uvicorn feature_store_service.main:app --reload
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

#### GET /api/v1/features/{symbol}
Get features for a specific symbol.

**Parameters:**
- `symbol` (path): The trading symbol
- `timeframe` (query): The timeframe (e.g., "1h", "1d")
- `start_date` (query): Start date in ISO format
- `end_date` (query): End date in ISO format
- `features` (query): Comma-separated list of features to retrieve

**Response:**
```json
{
  "symbol": "EUR/USD",
  "timeframe": "1h",
  "features": {
    "sma_20": [1.1012, 1.1015, 1.1020],
    "rsi_14": [45.2, 46.8, 48.3],
    "volatility": [0.0012, 0.0015, 0.0011]
  },
  "timestamps": ["2025-04-28T12:00:00Z", "2025-04-28T13:00:00Z", "2025-04-28T14:00:00Z"]
}
```

#### POST /api/v1/features/batch
Batch request for multiple features.

**Request Body:**
```json
{
  "symbols": ["EUR/USD", "GBP/USD"],
  "timeframe": "1h",
  "start_date": "2025-04-01T00:00:00Z",
  "end_date": "2025-04-28T00:00:00Z",
  "features": ["sma_20", "rsi_14", "macd"]
}
```

#### POST /api/v1/features/register
Register a new feature calculation.

**Request Body:**
```json
{
  "name": "custom_momentum",
  "description": "Custom momentum calculation",
  "parameters": {
    "window": 14,
    "smoothing": 3
  },
  "code": "def calculate(data, window=14, smoothing=3): ..."
}
```

## Indicator Implementation
The Feature Store Service is the canonical source for all indicator implementations in the platform. Indicators are implemented in the `indicators/` directory with the following structure:

- `indicators/trend.py`: Trend indicators (MA, MACD, etc.)
- `indicators/momentum.py`: Momentum indicators (RSI, Stochastic, etc.)
- `indicators/volatility.py`: Volatility indicators (Bollinger Bands, ATR, etc.)
- `indicators/volume.py`: Volume-based indicators

## Caching Strategy
The service implements a multi-level caching strategy:
1. In-memory cache for frequently accessed features
2. Redis cache for shared features across instances
3. Database for persistent storage

## Integration with Other Services
The Feature Store Service integrates with:

- Analysis Engine Service for providing indicator data
- ML Workbench Service for feature engineering
- Strategy Execution Engine for strategy evaluation

## Error Handling
The service uses standardized error responses from `common_lib.exceptions`.
