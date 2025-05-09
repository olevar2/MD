# Performance Testing Framework

This framework is used to run performance tests for the forex trading platform services and establish performance baselines.

## Overview

The performance testing framework includes:

- Load test scripts for different scenarios
- Test data for different market conditions
- Metrics collection and reporting
- Comparison with baselines
- Alerting on significant deviations

## Prerequisites

- Python 3.8+
- Locust (for load testing)
- Pandas (for data analysis)
- Matplotlib (for visualization)
- Prometheus (for metrics collection)

## Installation

```bash
pip install locust pandas matplotlib requests
```

## Directory Structure

```
performance_testing/
├── README.md
├── run_performance_tests.py
├── locustfile.py
├── baselines.json
├── data/
│   ├── normal_trading_data.csv
│   ├── market_open_data.csv
│   ├── high_volatility_data.csv
│   └── overnight_processing_data.csv
└── results/
    ├── analysis-engine-service/
    ├── trading-gateway-service/
    ├── feature-store-service/
    ├── ml-integration-service/
    ├── strategy-execution-engine/
    └── data-pipeline-service/
```

## Test Scenarios

The framework includes the following test scenarios:

1. **Normal Trading Hours**
   - Simulates normal trading hours with moderate market activity
   - Load: 100 requests per second
   - Duration: 30 minutes
   - Data: Historical data from a typical trading day

2. **Market Open**
   - Simulates market open with high market activity
   - Load: 500 requests per second
   - Duration: 15 minutes
   - Data: Historical data from market open periods

3. **High Volatility**
   - Simulates periods of high market volatility
   - Load: 300 requests per second
   - Duration: 30 minutes
   - Data: Historical data from high volatility periods

4. **Overnight Processing**
   - Simulates overnight batch processing
   - Load: 50 requests per second
   - Duration: 60 minutes
   - Data: Full day's historical data

## Usage

### Running Tests

To run performance tests for a specific service and scenario:

```bash
python run_performance_tests.py --service analysis-engine-service --scenario normal_trading
```

To run tests for all services and scenarios:

```bash
python run_performance_tests.py --all
```

### Updating Baselines

To update baselines with test results:

```bash
python run_performance_tests.py --service analysis-engine-service --scenario normal_trading --update-baseline
```

To update baselines for all services and scenarios:

```bash
python run_performance_tests.py --all --update-baseline
```

### Viewing Results

Test results are stored in the `results/` directory, organized by service and scenario. Each test run generates:

- CSV files with raw test data
- A performance report in Markdown format
- Visualizations of key metrics

## Metrics

The framework collects the following metrics:

- **Request Latency**: p50, p95, and p99 latency
- **Throughput**: Requests per second
- **Error Rate**: Percentage of failed requests
- **CPU Usage**: Average CPU usage during the test
- **Memory Usage**: Average memory usage during the test
- **Disk Usage**: Average disk usage during the test

## Baselines

Performance baselines are stored in `baselines.json` and are used to:

- Compare test results with established baselines
- Detect performance regressions
- Set appropriate alert thresholds

Baselines should be reviewed and updated quarterly or after significant system changes.

## Integration with Monitoring

The performance testing framework integrates with the monitoring system to:

- Use the same metrics collection mechanisms
- Compare test results with production metrics
- Update alert thresholds based on baselines

## Adding New Services

To add a new service to the performance testing framework:

1. Add the service to the `SERVICES` list in `run_performance_tests.py`
2. Add service endpoints to the `SERVICE_ENDPOINTS` dictionary in `locustfile.py`
3. Add request payloads for the service in the `get_payload` function in `locustfile.py`
4. Run tests to establish baselines for the new service

## Adding New Scenarios

To add a new test scenario:

1. Add the scenario to the `SCENARIOS` dictionary in `run_performance_tests.py`
2. Create a test data file in the `data/` directory
3. Run tests to establish baselines for the new scenario
