# Service Level Objectives (SLOs)

This document defines the Service Level Objectives (SLOs) and Service Level Indicators (SLIs) for the forex trading platform.

## trading_gateway_availability

**Description**: Trading Gateway API availability

**Service**: trading-gateway-service

### Service Level Indicator (SLI)

**Metric Name**: http_request_availability

**Metric Type**: availability

**Metric Query**:
```
sum(rate(http_requests_total{service="trading-gateway-service",status_code=~"[2345].."}[5m])) / sum(rate(http_requests_total{service="trading-gateway-service"}[5m]))
```

### Service Level Objective (SLO)

**Target**: 99.5%

**Window**: 30d

**Error Budget**: 0.5%

### Alerting

| Alert | Window | Burn Rate | Severity |
|-------|--------|-----------|----------|
| TradingGatewayHighErrorBudgetBurn1h | 1h | 14.4x | critical |
| TradingGatewayHighErrorBudgetBurn6h | 6h | 6x | warning |

## trading_gateway_latency

**Description**: Trading Gateway API latency

**Service**: trading-gateway-service

### Service Level Indicator (SLI)

**Metric Name**: http_request_latency

**Metric Type**: latency

**Metric Query**:
```
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{service="trading-gateway-service"}[5m])) by (le))
```

### Service Level Objective (SLO)

**Target**: 95.0%

**Threshold**: 200.0ms

**Window**: 30d

**Error Budget**: 5.0%

### Alerting

| Alert | Window | Burn Rate | Severity |
|-------|--------|-----------|----------|
| TradingGatewayHighLatencyBudgetBurn1h | 1h | 14.4x | critical |
| TradingGatewayHighLatencyBudgetBurn6h | 6h | 6x | warning |

## order_execution_success

**Description**: Order execution success rate

**Service**: trading-gateway-service

### Service Level Indicator (SLI)

**Metric Name**: order_execution_success

**Metric Type**: success_rate

**Metric Query**:
```
sum(rate(order_execution_success_total{service="trading-gateway-service"}[5m])) / sum(rate(order_execution_total{service="trading-gateway-service"}[5m]))
```

### Service Level Objective (SLO)

**Target**: 99.5%

**Window**: 30d

**Error Budget**: 0.5%

### Alerting

| Alert | Window | Burn Rate | Severity |
|-------|--------|-----------|----------|
| OrderExecutionHighErrorBudgetBurn1h | 1h | 14.4x | critical |
| OrderExecutionHighErrorBudgetBurn6h | 6h | 6x | warning |

## ml_model_inference_latency

**Description**: ML model inference latency

**Service**: ml-integration-service

### Service Level Indicator (SLI)

**Metric Name**: model_inference_latency

**Metric Type**: latency

**Metric Query**:
```
histogram_quantile(0.95, sum(rate(model_inference_duration_seconds_bucket{service="ml-integration-service"}[5m])) by (le))
```

### Service Level Objective (SLO)

**Target**: 95.0%

**Threshold**: 100.0ms

**Window**: 30d

**Error Budget**: 5.0%

### Alerting

| Alert | Window | Burn Rate | Severity |
|-------|--------|-----------|----------|
| MLInferenceHighLatencyBudgetBurn1h | 1h | 14.4x | critical |
| MLInferenceHighLatencyBudgetBurn6h | 6h | 6x | warning |

## strategy_execution_latency

**Description**: Strategy execution latency

**Service**: strategy-execution-engine

### Service Level Indicator (SLI)

**Metric Name**: strategy_execution_latency

**Metric Type**: latency

**Metric Query**:
```
histogram_quantile(0.95, sum(rate(strategy_execution_duration_seconds_bucket{service="strategy-execution-engine"}[5m])) by (le))
```

### Service Level Objective (SLO)

**Target**: 95.0%

**Threshold**: 500.0ms

**Window**: 30d

**Error Budget**: 5.0%

### Alerting

| Alert | Window | Burn Rate | Severity |
|-------|--------|-----------|----------|
| StrategyExecutionHighLatencyBudgetBurn1h | 1h | 14.4x | critical |
| StrategyExecutionHighLatencyBudgetBurn6h | 6h | 6x | warning |

