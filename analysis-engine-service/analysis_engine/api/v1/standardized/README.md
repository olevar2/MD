# Standardized API Endpoints

This document describes the standardized API endpoints for the Analysis Engine Service. All domains in the Analysis Engine Service have been successfully standardized following consistent patterns and best practices.

## API Design Principles

All standardized API endpoints follow these design principles:

1. **Consistent URL Structure**: `/api/v1/{service}/{domain}/{resource}`
   - `service`: The service name (e.g., `analysis`)
   - `domain`: The domain within the service (e.g., `adaptations`)
   - `resource`: The specific resource (e.g., `parameters`)

2. **Consistent HTTP Methods**:
   - `GET`: Retrieve resources
   - `POST`: Create resources or trigger actions
   - `PUT`: Update resources (full replacement)
   - `PATCH`: Partial update of resources
   - `DELETE`: Remove resources

3. **Standardized Response Format**:
   - Success responses use appropriate HTTP status codes (200, 201, 202, 204)
   - Error responses use appropriate HTTP status codes (400, 401, 403, 404, 500)
   - All responses include a consistent structure

4. **Comprehensive Documentation**:
   - All endpoints include summary and description
   - All parameters are documented
   - Example requests and responses are provided

5. **Backward Compatibility**:
   - Legacy endpoints are maintained for backward compatibility
   - Legacy endpoints include deprecation notices

## Available Endpoints

The following domains have been standardized:

1. **Adaptive Layer** (`/api/v1/analysis/adaptations/*`)
2. **Health Checks** (`/api/v1/analysis/health-checks/*`)
3. **Market Regime Analysis** (`/api/v1/analysis/market-regimes/*`)
4. **Signal Quality** (`/api/v1/analysis/signal-quality/*`)
5. **NLP Analysis** (`/api/v1/analysis/nlp/*`)
6. **Correlation Analysis** (`/api/v1/analysis/correlations/*`)
7. **Manipulation Detection** (`/api/v1/analysis/manipulation-detection/*`)
8. **Tool Effectiveness** (`/api/v1/analysis/effectiveness/*`)
9. **Feedback** (`/api/v1/analysis/feedback/*`)
10. **Monitoring** (`/api/v1/analysis/monitoring/*`)
11. **Causal Analysis** (`/api/v1/analysis/causal/*`)
12. **Backtesting** (`/api/v1/analysis/backtesting/*`)

### Adaptive Layer API

Base URL: `/api/v1/analysis/adaptations`

#### Parameters

- `POST /parameters/generate`: Generate adaptive parameters
- `POST /parameters/adjust`: Adjust strategy parameters
- `GET /parameters/history/{strategy_id}/{instrument}/{timeframe}`: Get parameter adjustment history

#### Strategy

- `POST /strategy/update`: Update strategy parameters
- `POST /strategy/recommendations`: Generate strategy recommendations
- `POST /strategy/effectiveness-trend`: Analyze strategy effectiveness trend

#### Feedback

- `POST /feedback/outcomes`: Record strategy outcome
- `GET /feedback/insights/{strategy_id}`: Get adaptation insights
- `GET /feedback/performance/{strategy_id}`: Get performance by regime

#### Adaptations

- `GET /adaptations/history`: Get adaptation history

### Health API

Base URL: `/api/v1/analysis/health-checks`

- `GET`: Get detailed health status
- `GET /liveness`: Liveness probe
- `GET /readiness`: Readiness probe

### Market Regime API

Base URL: `/api/v1/analysis/market-regimes`

- `GET /detect`: Detect market regime
- `POST /analyze`: Analyze market regime
- `GET /history/{instrument}/{timeframe}`: Get market regime history

### Signal Quality API

Base URL: `/api/v1/analysis/signal-quality`

- `POST /evaluate`: Evaluate signal quality
- `POST /filter`: Filter signals by quality
- `GET /metrics/{signal_id}`: Get signal quality metrics

### NLP Analysis API

Base URL: `/api/v1/analysis/nlp`

- `POST /sentiment`: Analyze sentiment
- `POST /extract-entities`: Extract entities
- `POST /summarize`: Summarize text

### Correlation Analysis API

Base URL: `/api/v1/analysis/correlations`

- `POST /analyze`: Analyze correlations
- `GET /matrix`: Get correlation matrix
- `POST /detect-changes`: Detect correlation changes

### Manipulation Detection API

Base URL: `/api/v1/analysis/manipulation-detection`

- `POST /detect`: Detect market manipulation
- `POST /analyze`: Analyze suspicious patterns
- `GET /alerts`: Get manipulation alerts

### Tool Effectiveness API

Base URL: `/api/v1/analysis/effectiveness`

- `POST /signals`: Register a signal
- `POST /outcomes`: Register an outcome
- `GET /metrics`: Get effectiveness metrics
- `GET /dashboard-data`: Get dashboard data

### Feedback API

Base URL: `/api/v1/analysis/feedback`

- `GET /statistics`: Get feedback statistics
- `POST /models/{model_id}/retrain`: Trigger model retraining
- `PUT /rules`: Update feedback rules
- `POST /submit`: Submit feedback

### Monitoring API

Base URL: `/api/v1/analysis/monitoring`

- `GET /async-performance`: Get async performance metrics
- `GET /memory`: Get memory metrics
- `POST /async-performance/report`: Trigger async performance report
- `GET /health`: Get service health

### Causal Analysis API

Base URL: `/api/v1/analysis/causal`

- `POST /discover-structure`: Discover causal structure
- `POST /estimate-effect`: Estimate causal effect
- `POST /counterfactual-analysis`: Perform counterfactual analysis
- `POST /currency-pair-relationships`: Analyze currency pair relationships

### Backtesting API

Base URL: `/api/v1/analysis/backtesting`

- `POST /run`: Run a backtest
- `GET /{backtest_id}`: Get backtest results
- `POST /walk-forward`: Run walk-forward optimization
- `POST /monte-carlo`: Run Monte Carlo simulation
- `POST /stress-test`: Run stress test

## Using the Standardized Clients

The Analysis Engine Service provides standardized clients for interacting with these API endpoints. All clients follow these patterns:

- **Resilience**: Retry with backoff and circuit breaking
- **Error Handling**: Consistent error handling with domain-specific exceptions
- **Logging**: Comprehensive structured logging
- **Timeout Handling**: Configurable timeouts for all operations
- **Type Hints**: Full type annotations for better developer experience

The following standardized clients are available:

1. **AdaptiveLayerClient**: Client for interacting with the Adaptive Layer API
2. **MarketRegimeClient**: Client for interacting with the Market Regime API
3. **SignalQualityClient**: Client for interacting with the Signal Quality API
4. **NLPAnalysisClient**: Client for interacting with the NLP Analysis API
5. **CorrelationAnalysisClient**: Client for interacting with the Correlation Analysis API
6. **ManipulationDetectionClient**: Client for interacting with the Manipulation Detection API
7. **EffectivenessClient**: Client for interacting with the Tool Effectiveness API
8. **FeedbackClient**: Client for interacting with the Feedback API
9. **MonitoringClient**: Client for interacting with the Monitoring API
10. **CausalClient**: Client for interacting with the Causal Analysis API
11. **BacktestingClient**: Client for interacting with the Backtesting API

### Example: Using the Adaptive Layer Client

```python
from analysis_engine.clients.standardized import get_client_factory

async def example():
    # Get the client factory
    factory = get_client_factory()

    # Get the adaptive layer client
    client = factory.get_adaptive_layer_client()

    # Generate adaptive parameters
    result = await client.generate_adaptive_parameters(
        strategy_id="macd_rsi_strategy_v1",
        symbol="EURUSD",
        timeframe="1h",
        ohlc_data=ohlc_data,
        available_tools=["macd", "rsi", "bollinger_bands"]
    )

    print(f"Generated parameters: {result}")
```

### Example: Using the Backtesting Client

```python
from analysis_engine.clients.standardized import get_client_factory
from datetime import datetime, timedelta

async def run_backtest_example():
    # Get the client factory
    factory = get_client_factory()

    # Get the backtesting client
    client = factory.get_backtesting_client()

    # Example strategy configuration
    strategy_config = {
        "strategy_id": "moving_average_crossover",
        "parameters": {
            "fast_period": 10,
            "slow_period": 30
        },
        "risk_settings": {
            "max_drawdown_pct": 20,
            "max_risk_per_trade_pct": 2
        }
    }

    # Run backtest
    result = await client.run_backtest(
        strategy_config=strategy_config,
        start_date=datetime.utcnow() - timedelta(days=365),
        end_date=datetime.utcnow(),
        instruments=["EUR_USD", "GBP_USD"],
        initial_capital=10000.0
    )

    print(f"Backtest completed with ID: {result.get('backtest_id')}")
```

## Migrating from Legacy Endpoints

Legacy endpoints are still available but will be deprecated in future versions. To migrate to the standardized endpoints:

1. Update your client code to use the standardized clients
2. Update your direct API calls to use the new URL structure
3. Test your application thoroughly

Legacy endpoints include deprecation notices in their documentation and log messages encouraging migration to the standardized endpoints.

## Validation

To validate that all API endpoints follow the standardization patterns, run the validation script:

```bash
python api_standardization_validator.py
```

This will generate a report of compliant and non-compliant endpoints.

## Documentation

For more detailed documentation on each domain, see the following:

- [API Standardization Plan](./API_STANDARDIZATION_PLAN.md)
- [Standardization Completion Report](./STANDARDIZATION_COMPLETION_REPORT.md)

## Next Steps

With the Analysis Engine Service standardization complete, the next steps are:

1. **Standardize Other Services**: Apply the same standardization patterns to other services
2. **Update Client Code**: Update client code to use the standardized clients
3. **Monitor Usage**: Monitor usage of legacy vs. standardized endpoints to track migration progress
4. **Documentation and Training**: Create comprehensive documentation and training materials
