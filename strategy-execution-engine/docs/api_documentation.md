# Strategy Execution Engine API Documentation

This document provides detailed information about the API endpoints available in the Strategy Execution Engine service.

## API Overview

The Strategy Execution Engine API follows the standardized API pattern for the Forex Trading Platform:

- All endpoints are prefixed with `/api/v1/[domain]/*`
- Responses follow a consistent format
- Error handling is standardized across all endpoints
- Authentication is required for all endpoints (except health checks)

## Base URL

The base URL for all API endpoints is:

```
http://{host}:{port}
```

Where:
- `{host}` is the hostname or IP address of the Strategy Execution Engine service
- `{port}` is the port number (default: 8003)

## Authentication

All API endpoints (except health checks) require authentication using an API key. The API key should be provided in the `X-API-Key` header.

Example:
```
X-API-Key: your-api-key
```

## Common Response Format

All API responses follow a consistent format:

### Success Response

```json
{
  "status": "success",
  "data": {
    // Response data specific to the endpoint
  },
  "metadata": {
    "timestamp": "2023-01-01T00:00:00Z",
    "version": "0.1.0"
  }
}
```

### Error Response

```json
{
  "status": "error",
  "error": {
    "code": "ERROR_CODE",
    "message": "Error message",
    "details": {
      // Additional error details
    }
  },
  "metadata": {
    "timestamp": "2023-01-01T00:00:00Z",
    "version": "0.1.0"
  }
}
```

## API Endpoints

### Root Endpoint

#### GET /

Returns basic information about the service.

**Response:**

```json
{
  "message": "Strategy Execution Engine is running",
  "version": "0.1.0",
  "timestamp": "2023-01-01T00:00:00Z"
}
```

### Health Check Endpoints

#### GET /health

Returns the health status of the service.

**Response:**

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2023-01-01T00:00:00Z"
}
```

#### GET /health/detailed

Returns detailed health status information, including the status of dependencies.

**Response:**

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2023-01-01T00:00:00Z",
  "database": {
    "status": "healthy",
    "message": "No database connection required"
  },
  "services": {
    "analysis_engine": {
      "status": "healthy",
      "message": "Connection successful"
    },
    "feature_store": {
      "status": "healthy",
      "message": "Connection successful"
    },
    "trading_gateway": {
      "status": "healthy",
      "message": "Connection successful"
    },
    "strategy_loader": {
      "status": "healthy",
      "message": "Strategy loader operational",
      "strategies_loaded": 5
    },
    "backtester": {
      "status": "healthy",
      "message": "Backtester operational"
    }
  }
}
```

### Strategy Endpoints

#### GET /api/v1/strategies

Returns a list of all registered strategies.

**Response:**

```json
{
  "strategies": [
    {
      "id": "strategy1",
      "name": "Moving Average Crossover",
      "type": "custom",
      "status": "active",
      "instruments": ["EUR/USD", "GBP/USD"],
      "timeframe": "1h",
      "description": "A simple moving average crossover strategy",
      "parameters": {
        "fast_period": 10,
        "slow_period": 20
      }
    },
    {
      "id": "strategy2",
      "name": "RSI Strategy",
      "type": "custom",
      "status": "active",
      "instruments": ["USD/JPY"],
      "timeframe": "4h",
      "description": "An RSI-based strategy",
      "parameters": {
        "rsi_period": 14,
        "overbought": 70,
        "oversold": 30
      }
    }
  ]
}
```

#### GET /api/v1/strategies/{strategy_id}

Returns details of a specific strategy.

**Parameters:**

- `strategy_id` (path): ID of the strategy to retrieve

**Response:**

```json
{
  "id": "strategy1",
  "name": "Moving Average Crossover",
  "type": "custom",
  "status": "active",
  "instruments": ["EUR/USD", "GBP/USD"],
  "timeframe": "1h",
  "description": "A simple moving average crossover strategy",
  "parameters": {
    "fast_period": 10,
    "slow_period": 20
  }
}
```

**Error Responses:**

- `404 Not Found`: Strategy not found

#### POST /api/v1/strategies/register

Registers a new strategy.

**Request Body:**

```json
{
  "name": "New Strategy",
  "description": "A new strategy",
  "instruments": ["EUR/USD", "GBP/USD"],
  "timeframe": "1h",
  "parameters": {
    "param1": 10,
    "param2": "value"
  },
  "code": "class NewStrategy(Strategy):\n    def analyze(self, data):\n        return {}\n"
}
```

**Response:**

```json
{
  "id": "new_strategy_id",
  "name": "New Strategy",
  "status": "active",
  "message": "Strategy New Strategy registered successfully"
}
```

**Error Responses:**

- `400 Bad Request`: Invalid strategy configuration
- `422 Unprocessable Entity`: Invalid request body

### Backtesting Endpoints

#### POST /api/v1/backtest

Runs a backtest for a strategy.

**Request Body:**

```json
{
  "strategy_id": "strategy1",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "initial_capital": 10000.0,
  "parameters": {
    "fast_period": 10,
    "slow_period": 20
  }
}
```

**Response:**

```json
{
  "backtest_id": "backtest1",
  "strategy_id": "strategy1",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "metrics": {
    "total_trades": 50,
    "winning_trades": 30,
    "losing_trades": 20,
    "win_rate": 0.6,
    "profit_factor": 1.5,
    "net_profit": 5000,
    "net_profit_pct": 50,
    "max_drawdown": 10
  },
  "trades": [
    {
      "id": "trade1",
      "position_id": "position1",
      "instrument": "EUR/USD",
      "type": "long",
      "entry_price": 1.1000,
      "entry_time": "2023-01-05T10:00:00",
      "exit_price": 1.1100,
      "exit_time": "2023-01-06T10:00:00",
      "size": 1.0,
      "profit_loss": 100,
      "profit_loss_pct": 1.0
    }
  ],
  "equity_curve": [
    {
      "timestamp": "2023-01-01T00:00:00",
      "equity": 10000
    },
    {
      "timestamp": "2023-12-31T00:00:00",
      "equity": 15000
    }
  ]
}
```

**Error Responses:**

- `400 Bad Request`: Invalid backtest configuration
- `404 Not Found`: Strategy not found
- `422 Unprocessable Entity`: Invalid request body

### Analysis Endpoints

#### GET /api/v1/analysis/backtests

Lists all backtests, optionally filtered by strategy ID.

**Parameters:**

- `strategy_id` (query, optional): Strategy ID to filter by

**Response:**

```json
{
  "backtests": [
    {
      "backtest_id": "backtest1",
      "strategy_id": "strategy1",
      "start_date": "2023-01-01",
      "end_date": "2023-12-31",
      "initial_capital": 10000.0,
      "metrics": {
        "total_trades": 50,
        "winning_trades": 30,
        "losing_trades": 20,
        "win_rate": 0.6,
        "profit_factor": 1.5,
        "net_profit": 5000,
        "net_profit_pct": 50,
        "max_drawdown": 10
      }
    },
    {
      "backtest_id": "backtest2",
      "strategy_id": "strategy2",
      "start_date": "2023-01-01",
      "end_date": "2023-12-31",
      "initial_capital": 10000.0,
      "metrics": {
        "total_trades": 40,
        "winning_trades": 25,
        "losing_trades": 15,
        "win_rate": 0.625,
        "profit_factor": 1.8,
        "net_profit": 6000,
        "net_profit_pct": 60,
        "max_drawdown": 8
      }
    }
  ]
}
```

#### GET /api/v1/analysis/performance/{backtest_id}

Analyzes the performance of a backtest.

**Parameters:**

- `backtest_id` (path): Backtest ID

**Response:**

```json
{
  "backtest_id": "backtest1",
  "strategy_id": "strategy1",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "metrics": {
    "total_trades": 50,
    "winning_trades": 30,
    "losing_trades": 20,
    "win_rate": 0.6,
    "profit_factor": 1.5,
    "net_profit": 5000,
    "net_profit_pct": 50,
    "max_drawdown": 10,
    "sharpe_ratio": 1.2,
    "sortino_ratio": 1.8,
    "calmar_ratio": 5.0
  },
  "trade_stats": {
    "avg_trade_duration": 12.5,
    "max_trade_duration": 48.0,
    "min_trade_duration": 1.0,
    "avg_profit": 250.0,
    "avg_loss": -150.0,
    "max_profit": 800.0,
    "max_loss": -400.0,
    "avg_profit_pct": 2.5,
    "avg_loss_pct": -1.5,
    "max_profit_pct": 8.0,
    "max_loss_pct": -4.0,
    "consecutive_wins": 5,
    "consecutive_losses": 3
  },
  "drawdown_stats": {
    "max_drawdown": 10.0,
    "avg_drawdown": 5.0,
    "top_drawdowns": [
      {
        "drawdown": 10.0,
        "peak_equity": 12000.0,
        "trough_equity": 10800.0,
        "peak_timestamp": "2023-06-01T00:00:00",
        "trough_timestamp": "2023-06-15T00:00:00",
        "recovery_time": 20.0,
        "recovery_timestamp": "2023-07-05T00:00:00"
      }
    ],
    "avg_recovery_time": 15.0,
    "max_recovery_time": 30.0,
    "avg_underwater_period": 10.0,
    "max_underwater_period": 25.0
  },
  "monthly_returns": {
    "2023-01": 5.0,
    "2023-02": 3.5,
    "2023-03": -2.0,
    "2023-04": 4.0,
    "2023-05": 6.5,
    "2023-06": -1.5,
    "2023-07": 3.0,
    "2023-08": 5.5,
    "2023-09": 4.0,
    "2023-10": 2.5,
    "2023-11": 3.0,
    "2023-12": 4.5
  }
}
```

**Error Responses:**

- `404 Not Found`: Backtest not found
- `500 Internal Server Error`: Analysis error

#### GET /api/v1/analysis/compare

Compares multiple strategies based on their backtest results.

**Parameters:**

- `strategy_ids` (query): List of strategy IDs to compare

**Response:**

```json
{
  "strategies": ["strategy1", "strategy2"],
  "metrics_comparison": {
    "win_rate": {
      "strategy1": 0.6,
      "strategy2": 0.625
    },
    "profit_factor": {
      "strategy1": 1.5,
      "strategy2": 1.8
    },
    "net_profit_pct": {
      "strategy1": 50.0,
      "strategy2": 60.0
    },
    "max_drawdown": {
      "strategy1": 10.0,
      "strategy2": 8.0
    },
    "sharpe_ratio": {
      "strategy1": 1.2,
      "strategy2": 1.5
    },
    "sortino_ratio": {
      "strategy1": 1.8,
      "strategy2": 2.2
    },
    "calmar_ratio": {
      "strategy1": 5.0,
      "strategy2": 7.5
    }
  },
  "trade_stats_comparison": {
    "avg_trade_duration": {
      "strategy1": 12.5,
      "strategy2": 10.0
    },
    "avg_profit": {
      "strategy1": 250.0,
      "strategy2": 300.0
    },
    "avg_loss": {
      "strategy1": -150.0,
      "strategy2": -120.0
    },
    "avg_profit_pct": {
      "strategy1": 2.5,
      "strategy2": 3.0
    },
    "avg_loss_pct": {
      "strategy1": -1.5,
      "strategy2": -1.2
    },
    "consecutive_wins": {
      "strategy1": 5,
      "strategy2": 6
    },
    "consecutive_losses": {
      "strategy1": 3,
      "strategy2": 2
    }
  },
  "drawdown_comparison": {
    "max_drawdown": {
      "strategy1": 10.0,
      "strategy2": 8.0
    },
    "avg_drawdown": {
      "strategy1": 5.0,
      "strategy2": 4.0
    },
    "avg_recovery_time": {
      "strategy1": 15.0,
      "strategy2": 12.0
    },
    "max_recovery_time": {
      "strategy1": 30.0,
      "strategy2": 25.0
    }
  },
  "timestamp": "2023-12-31T00:00:00Z"
}
```

**Error Responses:**

- `400 Bad Request`: No strategy IDs provided
- `500 Internal Server Error`: Comparison error

## Error Codes

The API uses the following error codes:

- `FOREX_PLATFORM_ERROR`: Generic platform error
- `STRATEGY_EXECUTION_ERROR`: Error during strategy execution
- `STRATEGY_CONFIGURATION_ERROR`: Invalid strategy configuration
- `STRATEGY_LOAD_ERROR`: Error loading strategy
- `BACKTEST_ERROR`: Generic backtest error
- `BACKTEST_CONFIG_ERROR`: Invalid backtest configuration
- `BACKTEST_DATA_ERROR`: Error with backtest data
- `BACKTEST_EXECUTION_ERROR`: Error during backtest execution
- `BACKTEST_REPORT_ERROR`: Error generating backtest report

## Rate Limiting

The API is rate-limited to prevent abuse. The rate limits are:

- 100 requests per minute per API key
- 1000 requests per hour per API key

If you exceed these limits, you will receive a `429 Too Many Requests` response.

## Versioning

The API is versioned using the URL path (e.g., `/api/v1/`). When a new version is released, the old version will be maintained for a period of time to allow for migration.

## Support

For support with the API, please contact the platform support team.
