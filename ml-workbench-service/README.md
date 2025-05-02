# ML Workbench Service

## Overview
The ML Workbench Service is a specialized environment for developing, training, and deploying machine learning models for forex trading applications. It provides a unified interface for data scientists and quants to experiment with ML-based trading strategies and integrate them with the platform's trading infrastructure.

## Setup

### Prerequisites
- Python 3.10 or higher
- Poetry (dependency management)
- GPU support (recommended for training)
- Access to feature data sources
- Network connectivity to other platform services

### Installation
1. Clone the repository
2. Navigate to the service directory:
```bash
cd ml-workbench-service
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
| `PORT` | Service port | 8005 |
| `FEATURE_STORE_URL` | URL to the Feature Store Service | http://localhost:8001 |
| `MODEL_REGISTRY_PATH` | Path to model registry storage | ./model_registry |
| `EXPERIMENT_TRACKING_URI` | MLflow tracking URI (if using MLflow) | http://localhost:5000 |
| `API_KEY` | API key for authentication | - |
| `MAX_TRAINING_MEMORY` | Maximum memory allocation for training (in GB) | 16 |
| `CUDA_VISIBLE_DEVICES` | GPU devices to use (if available) | 0,1 |

### Running the Service
Run the service using Poetry:
```bash
poetry run python -m ml_workbench_service.main
```

For development with auto-reload:
```bash
poetry run uvicorn ml_workbench_service.main:app --reload
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

#### GET /api/v1/models
List all available models.

**Response:**
```json
{
  "models": [
    {
      "id": "price_prediction_lstm_v2",
      "name": "LSTM Price Predictor",
      "type": "time_series",
      "version": "2.1.0",
      "status": "deployed",
      "created_at": "2025-03-12T10:15:22Z",
      "metrics": {
        "accuracy": 0.78,
        "mse": 0.0023
      }
    },
    {
      "id": "market_regime_classifier",
      "name": "Market Regime Classifier",
      "type": "classification",
      "version": "1.2.0",
      "status": "experimental",
      "created_at": "2025-04-05T16:30:45Z",
      "metrics": {
        "accuracy": 0.83,
        "f1_score": 0.81
      }
    }
  ]
}
```

#### POST /api/v1/models/train
Train a new model or retrain an existing model.

**Request Body:**
```json
{
  "name": "LSTM Price Predictor",
  "model_type": "time_series",
  "features": ["sma_20", "rsi_14", "macd", "bb_width"],
  "target": "price_change_1h",
  "symbols": ["EUR/USD", "GBP/USD"],
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2025-03-01T00:00:00Z",
  "hyperparameters": {
    "lstm_units": 64,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100
  },
  "validation_split": 0.2
}
```

#### POST /api/v1/models/{model_id}/predict
Get predictions from a model.

**Parameters:**
- `model_id` (path): The ID of the model

**Request Body:**
```json
{
  "symbol": "EUR/USD",
  "timeframe": "1h",
  "features": {
    "sma_20": [1.1012, 1.1015, 1.1020],
    "rsi_14": [45.2, 46.8, 48.3],
    "macd": [0.0012, 0.0015, 0.0018],
    "bb_width": [0.0025, 0.0024, 0.0028]
  }
}
```

#### POST /api/v1/models/{model_id}/deploy
Deploy a model to production.

**Parameters:**
- `model_id` (path): The ID of the model

**Request Body:**
```json
{
  "version": "2.1.0",
  "deployment_target": "strategy_engine",
  "scaling_factor": 1.0,
  "monitoring_level": "high"
}
```

## Model Types
The ML Workbench Service supports various model types:

1. **Time Series Prediction**: LSTM, GRU, and transformer models for price prediction
2. **Classification**: Models for market regime classification, trend direction, etc.
3. **Reinforcement Learning**: RL agents for optimizing trading decisions
4. **Anomaly Detection**: Models for detecting market anomalies
5. **Feature Engineering**: Automated feature selection and generation

## Experiment Tracking
The service integrates with MLflow for experiment tracking, providing:
- Hyperparameter tracking
- Model versioning
- Metric logging
- Artifact storage
- Experiment comparison

## Integration with Other Services
The ML Workbench Service integrates with:

- Feature Store Service for feature data
- Strategy Execution Engine for deploying ML-enhanced strategies
- ML Integration Service for API access to models
- Analysis Engine Service for enhanced market analysis

## Error Handling
The service uses standardized error responses from `common_lib.exceptions`.
