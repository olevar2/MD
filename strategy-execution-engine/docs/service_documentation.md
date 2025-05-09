# Strategy Execution Engine Service Documentation

This document provides detailed information about the Strategy Execution Engine service, including its architecture, components, and integration with other services.

## Service Overview

The Strategy Execution Engine is responsible for executing trading strategies and performing backtesting for the Forex Trading Platform. It provides a standardized API for strategy registration, execution, and backtesting.

## Architecture

The Strategy Execution Engine follows a clean architecture with the following layers:

1. **API Layer**: Handles HTTP requests and responses
2. **Service Layer**: Contains business logic for strategy execution and backtesting
3. **Domain Layer**: Contains domain models and business rules
4. **Infrastructure Layer**: Handles external dependencies and technical concerns

### Components

The service consists of the following main components:

- **Strategy Loader**: Loads and manages trading strategies
- **Backtester**: Performs backtesting of strategies
- **Service Container**: Manages dependencies and service lifecycle
- **Error Handling**: Provides standardized error handling
- **Monitoring**: Collects metrics and provides health checks

## Configuration

The service is configured using environment variables and configuration files. The following environment variables are supported:

- `DEBUG_MODE`: Enable debug mode (default: `false`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `HOST`: Host to bind to (default: `0.0.0.0`)
- `PORT`: Port to listen on (default: `8003`)
- `API_KEY`: API key for authentication
- `SERVICE_API_KEY`: API key for service-to-service communication
- `ANALYSIS_ENGINE_URL`: URL of the Analysis Engine service
- `FEATURE_STORE_URL`: URL of the Feature Store service
- `TRADING_GATEWAY_URL`: URL of the Trading Gateway service
- `RISK_MANAGEMENT_URL`: URL of the Risk Management service
- `PORTFOLIO_MANAGEMENT_URL`: URL of the Portfolio Management service
- `MONITORING_SERVICE_URL`: URL of the Monitoring & Alerting service
- `STRATEGIES_DIR`: Directory containing strategy implementations
- `BACKTEST_DATA_DIR`: Directory for storing backtest data

## Deployment

The service can be deployed using Docker or Kubernetes. A Dockerfile is provided for building a container image.

### Docker Deployment

```bash
# Build the image
docker build -t strategy-execution-engine .

# Run the container
docker run -p 8003:8003 \
  -e API_KEY=your-api-key \
  -e ANALYSIS_ENGINE_URL=http://analysis-engine-service:8002 \
  -e FEATURE_STORE_URL=http://feature-store-service:8001 \
  -e TRADING_GATEWAY_URL=http://trading-gateway-service:8004 \
  strategy-execution-engine
```

### Kubernetes Deployment

A Kubernetes deployment manifest is provided in the `k8s` directory.

```bash
kubectl apply -f k8s/deployment.yaml
```

## Integration with Other Services

The Strategy Execution Engine integrates with the following services:

### Analysis Engine Service

The Analysis Engine service provides market analysis and technical indicators used by strategies. The Strategy Execution Engine calls the Analysis Engine API to get analysis results.

Integration points:
- Getting technical indicators
- Getting market regime analysis
- Getting pattern recognition results

### Feature Store Service

The Feature Store service provides feature data for machine learning models used by strategies. The Strategy Execution Engine calls the Feature Store API to get feature data.

Integration points:
- Getting feature data for strategies
- Getting historical data for backtesting

### Trading Gateway Service

The Trading Gateway service provides access to trading platforms and brokers. The Strategy Execution Engine calls the Trading Gateway API to execute trades.

Integration points:
- Executing trades
- Getting account information
- Getting market data

### Risk Management Service

The Risk Management service provides risk management functionality. The Strategy Execution Engine calls the Risk Management API to check risk limits.

Integration points:
- Checking risk limits
- Getting risk parameters

### Portfolio Management Service

The Portfolio Management service provides portfolio management functionality. The Strategy Execution Engine calls the Portfolio Management API to get portfolio information.

Integration points:
- Getting portfolio information
- Getting allocation limits

### Monitoring & Alerting Service

The Monitoring & Alerting service provides monitoring and alerting functionality. The Strategy Execution Engine sends metrics and logs to the Monitoring & Alerting service.

Integration points:
- Sending metrics
- Sending logs
- Sending alerts

## Error Handling

The service uses a standardized error handling approach with custom exceptions and error responses. All errors are logged with appropriate context and returned to the client with a consistent format.

### Exception Hierarchy

- `ForexTradingPlatformError`: Base exception for all platform errors
  - `StrategyExecutionError`: Base exception for strategy execution errors
    - `StrategyConfigurationError`: Invalid strategy configuration
    - `StrategyLoadError`: Error loading strategy
  - `BacktestError`: Base exception for backtest errors
    - `BacktestConfigError`: Invalid backtest configuration
    - `BacktestDataError`: Error with backtest data
    - `BacktestExecutionError`: Error during backtest execution
    - `BacktestReportError`: Error generating backtest report

## Monitoring

The service provides monitoring through Prometheus metrics and health checks. The following metrics are collected:

- HTTP request count and duration
- Strategy execution count and duration
- Backtest execution count and duration
- Active strategies count

Health checks are available at:
- `/health`: Basic health check
- `/health/detailed`: Detailed health check with dependency status

## Logging

The service uses structured logging with JSON format. Logs include the following information:

- Timestamp
- Log level
- Logger name
- Message
- Module
- Function
- Line number
- Process ID
- Thread ID
- Exception information (if applicable)
- Request ID (for API requests)
- Additional context

## Testing

The service includes comprehensive unit and integration tests. Tests are organized by component and can be run using pytest.

```bash
# Run all tests
pytest

# Run unit tests only
pytest -m unit

# Run integration tests only
pytest -m integration

# Run with coverage
pytest --cov=strategy_execution_engine
```

## Development

### Prerequisites

- Python 3.9 or higher
- pip
- virtualenv (optional)

### Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### Running the Service

```bash
# Run the service
python -m strategy_execution_engine.main

# Run with custom port
PORT=8080 python -m strategy_execution_engine.main

# Run in debug mode
DEBUG_MODE=true python -m strategy_execution_engine.main
```

### Code Style

The service follows the PEP 8 style guide. Code style is enforced using flake8 and black.

```bash
# Check code style
flake8 strategy_execution_engine

# Format code
black strategy_execution_engine
```

## Support

For support with the service, please contact the platform support team.
