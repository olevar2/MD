# Strategy Execution Engine

## Overview
The Strategy Execution Engine is a critical component of the Forex Trading Platform responsible for executing trading strategies based on market data and analytical signals. It provides a framework for strategy definition, backtesting, optimization, and live execution.

## Setup

### Prerequisites
- Python 3.10 or higher
- Poetry (dependency management)
- Access to market data sources
- Network connectivity to other platform services

### Installation
1. Clone the repository
2. Navigate to the service directory:
```bash
cd strategy-execution-engine
```
3. Install dependencies using Poetry:
```bash
poetry install
```

### Environment Variables
The following environment variables are required (see `.env.example` for a complete list):

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `PORT` | Service port | 8003 |
| `ANALYSIS_ENGINE_URL` | URL to the Analysis Engine Service | http://analysis-engine-service:8002 |
| `FEATURE_STORE_URL` | URL to the Feature Store Service | http://feature-store-service:8001 |
| `TRADING_GATEWAY_URL` | URL to the Trading Gateway Service | http://trading-gateway-service:8004 |
| `RISK_MANAGEMENT_URL` | URL to the Risk Management Service | http://risk-management-service:8000 |
| `PORTFOLIO_MANAGEMENT_URL` | URL to the Portfolio Management Service | http://portfolio-management-service:8000 |
| `API_KEY` | API key for authentication | - |
| `SERVICE_API_KEY` | API key for this service | - |
| `TRADING_GATEWAY_KEY` | API key for trading gateway | - |
| `ANALYSIS_ENGINE_KEY` | API key for analysis engine | - |
| `FEATURE_STORE_KEY` | API key for feature store | - |
| `REDIS_URL` | Redis connection string for caching | redis://localhost:6379 |
| `DATABASE_URL` | Database connection string | postgresql://user:password@localhost:5432/strategy_execution |
| `MAX_CONCURRENT_STRATEGIES` | Maximum number of concurrent strategies | 10 |

For local development:
1. Copy `.env.example` to `.env` 
2. Update all sensitive values with your actual credentials
3. Never commit `.env` files to version control

### Running the Service
Run the service using Poetry:
```bash
poetry run python -m strategy_execution_engine.main
```

For development with auto-reload:
```bash
poetry run uvicorn strategy_execution_engine.main:app --reload
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

#### GET /api/v1/strategies
List all registered strategies.

**Response:**
```json
{
  "strategies": [
    {
      "id": "trend_following_v1",
      "name": "Trend Following Strategy",
      "type": "momentum",
      "status": "active",
      "instruments": ["EUR/USD", "GBP/USD"],
      "timeframe": "1h"
    },
    {
      "id": "causal_enhanced_v2",
      "name": "Causal Enhanced Strategy",
      "type": "ml_enhanced",
      "status": "backtesting",
      "instruments": ["EUR/USD"],
      "timeframe": "4h"
    }
  ]
}
```

#### POST /api/v1/strategies/register
Register a new strategy.

**Request Body:**
```json
{
  "name": "Custom Mean Reversion",
  "description": "Mean reversion strategy with dynamic thresholds",
  "instruments": ["EUR/USD", "USD/JPY"],
  "timeframe": "1h",
  "parameters": {
    "lookback_period": 20,
    "entry_threshold": 2.0,
    "exit_threshold": 0.5
  },
  "code": "class MeanReversionStrategy(BaseStrategy): ..."
}
```

#### POST /api/v1/backtest
Run a backtest for a strategy.

**Request Body:**
```json
{
  "strategy_id": "trend_following_v1",
  "start_date": "2025-01-01T00:00:00Z",
  "end_date": "2025-04-01T00:00:00Z",
  "initial_capital": 10000,
  "parameters": {
    "fast_period": 12,
    "slow_period": 26
  }
}
```

#### POST /api/v1/execute
Execute a strategy in live mode.

**Request Body:**
```json
{
  "strategy_id": "trend_following_v1",
  "capital_allocation": 5000,
  "parameters": {
    "fast_period": 12,
    "slow_period": 26
  }
}
```

## Strategy Types
The Strategy Execution Engine supports various strategy types:

1. **Trend Following**: Based on moving averages and trend indicators
2. **Mean Reversion**: Based on overbought/oversold conditions
3. **Breakout**: Based on price breaking support/resistance levels
4. **Causal Enhanced**: Strategies using causal inference and enhanced analytics
5. **ML-Enhanced**: Strategies incorporating machine learning predictions

## CausalEnhancedStrategy
A key feature of the engine is the CausalEnhancedStrategy framework that leverages causal inference techniques to improve decision-making. This strategy uses the Analysis Engine API for enhanced market data.

## Integration with Other Services
The Strategy Execution Engine integrates with:

- Analysis Engine Service for market analysis
- Feature Store Service for indicator data
- Trading Gateway Service for order execution
- Portfolio Management Service for capital allocation
- Risk Management Service for risk controls

## Error Handling
The service uses standardized error responses from `common_lib.exceptions` and implements retry logic using `common_lib.resilience.retry_with_policy`.

## Security
- Authentication is handled using API keys through `common_lib.security`
- All sensitive configuration is loaded via environment variables
- No hardcoded secrets in the codebase
- Communication between services uses secure channels with proper authentication
- API keys for integrated services are stored securely in environment variables
- Redis connections support authentication via password in REDIS_URL or separate REDIS_PASSWORD
