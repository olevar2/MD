# User Guide

This guide provides detailed instructions on how to use the Forex Trading Platform.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Market Data](#market-data)
3. [Technical Analysis](#technical-analysis)
4. [Machine Learning Models](#machine-learning-models)
5. [Portfolio Management](#portfolio-management)
6. [Trading](#trading)
7. [Monitoring and Alerting](#monitoring-and-alerting)
8. [Advanced Features](#advanced-features)

## Getting Started

### Platform Overview

The Forex Trading Platform is a comprehensive system for forex trading, analysis, and portfolio management. It consists of several services that work together to provide a complete trading solution:

- **Data Pipeline Service**: Collects and processes market data
- **Feature Store Service**: Manages features for analysis and machine learning
- **Analysis Engine Service**: Performs technical analysis and pattern recognition
- **ML Integration Service**: Manages machine learning models and predictions
- **Trading Gateway Service**: Connects to brokers and executes trades
- **Portfolio Management Service**: Manages trading accounts and positions
- **Monitoring Alerting Service**: Monitors system health and sends alerts

### Accessing the Platform

The platform can be accessed through:

1. **REST APIs**: Each service provides a REST API for programmatic access
2. **Command Line Interface**: Scripts for common operations
3. **Web Interface**: (If implemented) A web interface for user-friendly access

### Authentication

To access the platform APIs, you need to authenticate:

```python
import requests

# Example authentication
api_key = "your_api_key"
headers = {"Authorization": f"Bearer {api_key}"}

# Make authenticated request
response = requests.get(
    "http://localhost:8001/api/v1/symbols",
    headers=headers
)
```

## Market Data

### Accessing Market Data

The Data Pipeline Service provides access to market data:

```python
import requests

# Get EURUSD OHLCV data
response = requests.get(
    "http://localhost:8001/api/v1/historical-data",
    params={
        "symbol": "EURUSD",
        "timeframe": "1h",
        "start": "2023-01-01T00:00:00Z",
        "end": "2023-01-31T23:59:59Z"
    }
)

data = response.json()
print(data)
```

### Available Symbols

To get a list of available symbols:

```python
import requests

# Get all symbols
response = requests.get("http://localhost:8001/api/v1/symbols")
symbols = response.json()

# Filter forex symbols
forex_symbols = [s for s in symbols if s["type"] == "forex"]
print(forex_symbols)
```

### Available Timeframes

The platform supports the following timeframes:

- `1m`: 1 minute
- `5m`: 5 minutes
- `15m`: 15 minutes
- `30m`: 30 minutes
- `1h`: 1 hour
- `4h`: 4 hours
- `1d`: 1 day
- `1w`: 1 week
- `1M`: 1 month

### Downloading Historical Data

To download historical data for offline analysis:

```python
import requests

# Download EURUSD historical data as CSV
response = requests.get(
    "http://localhost:8001/api/v1/historical-data/download",
    params={
        "symbol": "EURUSD",
        "timeframe": "1h",
        "start": "2023-01-01T00:00:00Z",
        "end": "2023-01-31T23:59:59Z",
        "format": "csv"
    }
)

with open("eurusd_1h.csv", "wb") as f:
    f.write(response.content)
```

## Technical Analysis

### Calculating Indicators

The Analysis Engine Service provides technical indicators:

```python
import requests

# Calculate RSI for EURUSD
response = requests.post(
    "http://localhost:8003/api/v1/indicators/calculate",
    json={
        "indicator": "rsi",
        "symbol": "EURUSD",
        "timeframe": "1h",
        "parameters": {
            "period": 14
        },
        "start": "2023-01-01T00:00:00Z",
        "end": "2023-01-31T23:59:59Z"
    }
)

data = response.json()
print(data)
```

### Available Indicators

The platform supports a wide range of technical indicators:

- **Trend Indicators**: Moving Averages, MACD, ADX, Parabolic SAR
- **Momentum Indicators**: RSI, Stochastic, CCI, Williams %R
- **Volatility Indicators**: Bollinger Bands, ATR, Standard Deviation
- **Volume Indicators**: OBV, Volume, Money Flow Index
- **Custom Indicators**: User-defined indicators

To get a list of available indicators:

```python
import requests

# Get all indicators
response = requests.get("http://localhost:8003/api/v1/indicators")
indicators = response.json()
print(indicators)
```

### Pattern Recognition

The Analysis Engine Service can identify chart patterns:

```python
import requests

# Identify patterns for EURUSD
response = requests.post(
    "http://localhost:8003/api/v1/patterns/identify",
    json={
        "symbol": "EURUSD",
        "timeframe": "1h",
        "start": "2023-01-01T00:00:00Z",
        "end": "2023-01-31T23:59:59Z",
        "patterns": ["head_and_shoulders", "double_top", "double_bottom"]
    }
)

patterns = response.json()
print(patterns)
```

### Support and Resistance Levels

To identify support and resistance levels:

```python
import requests

# Get support and resistance levels for EURUSD
response = requests.post(
    "http://localhost:8003/api/v1/support-resistance",
    json={
        "symbol": "EURUSD",
        "timeframe": "1h",
        "start": "2023-01-01T00:00:00Z",
        "end": "2023-01-31T23:59:59Z",
        "method": "price_action"
    }
)

levels = response.json()
print(levels)
```

## Machine Learning Models

### Available Models

The ML Integration Service provides machine learning models for market prediction:

```python
import requests

# Get all models
response = requests.get("http://localhost:8004/api/v1/models")
models = response.json()
print(models)
```

### Making Predictions

To make predictions using a model:

```python
import requests

# Make prediction using a model
response = requests.post(
    "http://localhost:8004/api/v1/models/predict",
    json={
        "model_id": "price_direction_lstm",
        "symbol": "EURUSD",
        "timeframe": "1h",
        "features": ["close", "volume", "rsi_14", "macd_12_26_9"],
        "horizon": 24  # predict 24 hours ahead
    }
)

prediction = response.json()
print(prediction)
```

### Training Custom Models

To train a custom model:

```python
import requests

# Train a custom model
response = requests.post(
    "http://localhost:8004/api/v1/models/train",
    json={
        "name": "my_custom_model",
        "model_type": "lstm",
        "symbol": "EURUSD",
        "timeframe": "1h",
        "features": ["close", "volume", "rsi_14", "macd_12_26_9"],
        "target": "close",
        "target_transform": "pct_change",
        "train_start": "2020-01-01T00:00:00Z",
        "train_end": "2022-12-31T23:59:59Z",
        "test_start": "2023-01-01T00:00:00Z",
        "test_end": "2023-01-31T23:59:59Z",
        "parameters": {
            "units": 64,
            "dropout": 0.2,
            "epochs": 100,
            "batch_size": 32
        }
    }
)

training_job = response.json()
print(training_job)
```

## Portfolio Management

### Managing Accounts

The Portfolio Management Service manages trading accounts:

```python
import requests

# Get all accounts
response = requests.get("http://localhost:8006/api/v1/accounts")
accounts = response.json()
print(accounts)

# Get account details
account_id = accounts[0]["id"]
response = requests.get(f"http://localhost:8006/api/v1/accounts/{account_id}")
account = response.json()
print(account)
```

### Managing Positions

To manage trading positions:

```python
import requests

# Get all positions for an account
account_id = "ACC001"
response = requests.get(
    f"http://localhost:8006/api/v1/accounts/{account_id}/positions"
)
positions = response.json()
print(positions)

# Get position details
position_id = positions[0]["id"]
response = requests.get(
    f"http://localhost:8006/api/v1/positions/{position_id}"
)
position = response.json()
print(position)
```

### Performance Analysis

To analyze trading performance:

```python
import requests

# Get performance metrics for an account
account_id = "ACC001"
response = requests.get(
    f"http://localhost:8006/api/v1/accounts/{account_id}/performance",
    params={
        "start": "2023-01-01T00:00:00Z",
        "end": "2023-01-31T23:59:59Z",
        "metrics": ["profit_loss", "win_rate", "sharpe_ratio", "drawdown"]
    }
)
performance = response.json()
print(performance)
```

## Trading

### Placing Orders

The Trading Gateway Service handles order execution:

```python
import requests

# Place a market order
response = requests.post(
    "http://localhost:8005/api/v1/orders",
    json={
        "account_id": "ACC001",
        "symbol": "EURUSD",
        "order_type": "market",
        "direction": "buy",
        "volume": 0.1,
        "stop_loss": 1.0750,
        "take_profit": 1.0850
    }
)

order = response.json()
print(order)
```

### Order Types

The platform supports the following order types:

- **Market Order**: Execute immediately at the current market price
- **Limit Order**: Execute when the price reaches a specified level
- **Stop Order**: Execute when the price crosses a specified level
- **Stop-Limit Order**: Combine stop and limit orders

### Managing Orders

To manage existing orders:

```python
import requests

# Get all orders for an account
account_id = "ACC001"
response = requests.get(
    f"http://localhost:8005/api/v1/accounts/{account_id}/orders"
)
orders = response.json()
print(orders)

# Cancel an order
order_id = orders[0]["id"]
response = requests.delete(
    f"http://localhost:8005/api/v1/orders/{order_id}"
)
result = response.json()
print(result)
```

### Trading Strategies

To execute trading strategies:

```python
import requests

# Execute a trading strategy
response = requests.post(
    "http://localhost:8005/api/v1/strategies/execute",
    json={
        "strategy_id": "moving_average_crossover",
        "account_id": "ACC001",
        "symbol": "EURUSD",
        "parameters": {
            "fast_period": 10,
            "slow_period": 20,
            "volume": 0.1
        }
    }
)

execution = response.json()
print(execution)
```

## Monitoring and Alerting

### System Health

The Monitoring Alerting Service provides system health information:

```python
import requests

# Get system health
response = requests.get("http://localhost:8007/api/v1/health")
health = response.json()
print(health)
```

### Setting Alerts

To set up alerts:

```python
import requests

# Create a price alert
response = requests.post(
    "http://localhost:8007/api/v1/alerts",
    json={
        "name": "EURUSD Price Alert",
        "type": "price",
        "symbol": "EURUSD",
        "condition": "above",
        "threshold": 1.0800,
        "notification_channels": ["email"],
        "message": "EURUSD price is above 1.0800"
    }
)

alert = response.json()
print(alert)
```

### Alert Types

The platform supports the following alert types:

- **Price Alerts**: Trigger when price crosses a threshold
- **Indicator Alerts**: Trigger when an indicator crosses a threshold
- **Pattern Alerts**: Trigger when a pattern is detected
- **Performance Alerts**: Trigger when performance metrics cross thresholds
- **System Alerts**: Trigger when system metrics cross thresholds

## Advanced Features

### Backtesting

To backtest trading strategies:

```python
import requests

# Backtest a strategy
response = requests.post(
    "http://localhost:8003/api/v1/backtest",
    json={
        "strategy_id": "moving_average_crossover",
        "symbol": "EURUSD",
        "timeframe": "1h",
        "start": "2022-01-01T00:00:00Z",
        "end": "2022-12-31T23:59:59Z",
        "initial_capital": 10000,
        "parameters": {
            "fast_period": 10,
            "slow_period": 20,
            "volume": 0.1
        }
    }
)

backtest_result = response.json()
print(backtest_result)
```

### Optimization

To optimize strategy parameters:

```python
import requests

# Optimize strategy parameters
response = requests.post(
    "http://localhost:8003/api/v1/optimize",
    json={
        "strategy_id": "moving_average_crossover",
        "symbol": "EURUSD",
        "timeframe": "1h",
        "start": "2022-01-01T00:00:00Z",
        "end": "2022-12-31T23:59:59Z",
        "initial_capital": 10000,
        "parameters": {
            "fast_period": {"min": 5, "max": 20, "step": 1},
            "slow_period": {"min": 20, "max": 50, "step": 5},
            "volume": 0.1
        },
        "optimization_metric": "sharpe_ratio",
        "optimization_method": "grid_search"
    }
)

optimization_result = response.json()
print(optimization_result)
```

### Custom Indicators

To create custom indicators:

```python
import requests

# Create a custom indicator
response = requests.post(
    "http://localhost:8003/api/v1/indicators/custom",
    json={
        "name": "my_custom_indicator",
        "description": "My custom indicator",
        "formula": "ta.sma(close, 10) - ta.sma(close, 20)",
        "parameters": {
            "fast_period": 10,
            "slow_period": 20
        }
    }
)

custom_indicator = response.json()
print(custom_indicator)
```

### Data Export

To export data for external analysis:

```python
import requests

# Export data
response = requests.get(
    "http://localhost:8001/api/v1/export",
    params={
        "symbol": "EURUSD",
        "timeframe": "1h",
        "start": "2023-01-01T00:00:00Z",
        "end": "2023-01-31T23:59:59Z",
        "indicators": ["sma_10", "sma_20", "rsi_14"],
        "format": "csv"
    }
)

with open("eurusd_1h_with_indicators.csv", "wb") as f:
    f.write(response.content)
```
