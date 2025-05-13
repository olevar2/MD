# Error Handling and Resilience Analysis Report

## Summary

- Total Error Types: 185
- Total Error Handlers: 94
- Total Resilience Patterns: 14
- Total Error Handling Patterns: 820
- Total Resilience Usage: 188

## Error Handling Coverage

| Service | Total Patterns | Custom Error Usage | Standard Error Usage | Custom Error Percentage |
|---------|---------------|-------------------|---------------------|-------------------------|
| feature-store-service | 77 | 17 | 60 | 22.08% |
| ui-service | 0 | 0 | 0 | 0.00% |
| strategy-execution-engine | 53 | 24 | 29 | 45.28% |
| trading-gateway-service | 37 | 16 | 21 | 43.24% |
| common-js-lib | 0 | 0 | 0 | 0.00% |
| analysis-engine-service | 286 | 87 | 199 | 30.42% |
| common-lib | 44 | 9 | 35 | 20.45% |
| risk-management-service | 14 | 2 | 12 | 14.29% |
| core-foundations | 43 | 6 | 37 | 13.95% |
| feature_store_service | 0 | 0 | 0 | 0.00% |
| ml-integration-service | 26 | 18 | 8 | 69.23% |
| analysis_engine | 1 | 0 | 1 | 0.00% |
| model-registry-service | 11 | 11 | 0 | 100.00% |
| portfolio-management-service | 15 | 8 | 7 | 53.33% |
| monitoring-alerting-service | 23 | 3 | 20 | 13.04% |
| ml_workbench-service | 127 | 41 | 86 | 32.28% |
| api-gateway | 1 | 0 | 1 | 0.00% |
| data-management-service | 17 | 0 | 17 | 0.00% |
| data-pipeline-service | 45 | 35 | 10 | 77.78% |

## Resilience Coverage

| Service | Total Usage | Circuit Breaker | Retry | Timeout | Bulkhead | Fallback |
|---------|-------------|----------------|-------|---------|----------|----------|
| feature-store-service | 0 | 0 | 0 | 0 | 0 | 0 |
| ui-service | 0 | 0 | 0 | 0 | 0 | 0 |
| strategy-execution-engine | 10 | 1 | 9 | 0 | 0 | 0 |
| trading-gateway-service | 4 | 1 | 3 | 0 | 0 | 0 |
| common-js-lib | 0 | 0 | 0 | 0 | 0 | 0 |
| analysis-engine-service | 62 | 25 | 24 | 3 | 3 | 7 |
| common-lib | 77 | 21 | 32 | 7 | 17 | 0 |
| risk-management-service | 7 | 4 | 3 | 0 | 0 | 0 |
| core-foundations | 15 | 7 | 7 | 0 | 0 | 1 |
| feature_store_service | 0 | 0 | 0 | 0 | 0 | 0 |
| ml-integration-service | 2 | 2 | 0 | 0 | 0 | 0 |
| analysis_engine | 0 | 0 | 0 | 0 | 0 | 0 |
| model-registry-service | 0 | 0 | 0 | 0 | 0 | 0 |
| portfolio-management-service | 3 | 0 | 3 | 0 | 0 | 0 |
| monitoring-alerting-service | 0 | 0 | 0 | 0 | 0 | 0 |
| ml_workbench-service | 0 | 0 | 0 | 0 | 0 | 0 |
| api-gateway | 8 | 4 | 4 | 0 | 0 | 0 |
| data-management-service | 0 | 0 | 0 | 0 | 0 | 0 |
| data-pipeline-service | 0 | 0 | 0 | 0 | 0 | 0 |

## Error Handling Issues

| Service | Issue |
|---------|-------|
| feature-store-service | Low custom error usage (22.08%) |
| ui-service | No error handling patterns found |
| strategy-execution-engine | Low custom error usage (45.28%) |
| trading-gateway-service | Low custom error usage (43.24%) |
| common-js-lib | No error handling patterns found |
| analysis-engine-service | Low custom error usage (30.42%) |
| risk-management-service | Low custom error usage (14.29%) |
| feature_store_service | No error handling patterns found |
| analysis_engine | Low custom error usage (0.00%) |
| monitoring-alerting-service | Low custom error usage (13.04%) |
| ml_workbench-service | Low custom error usage (32.28%) |
| api-gateway | Low custom error usage (0.00%) |
| data-management-service | Low custom error usage (0.00%) |

## Resilience Issues

| Service | Issue |
|---------|-------|
| feature-store-service | No resilience patterns used |
| ui-service | No resilience patterns used |
| strategy-execution-engine | Missing resilience patterns: fallback, bulkhead, timeout |
| trading-gateway-service | Missing resilience patterns: fallback, bulkhead, timeout |
| common-js-lib | No resilience patterns used |
| risk-management-service | Missing resilience patterns: fallback, bulkhead, timeout |
| feature_store_service | No resilience patterns used |
| ml-integration-service | Missing resilience patterns: retry, bulkhead, timeout, fallback |
| analysis_engine | No resilience patterns used |
| model-registry-service | No resilience patterns used |
| portfolio-management-service | Missing resilience patterns: fallback, bulkhead, timeout, circuit_breaker |
| monitoring-alerting-service | No resilience patterns used |
| ml_workbench-service | No resilience patterns used |
| api-gateway | Missing resilience patterns: fallback, bulkhead, timeout |
| data-management-service | No resilience patterns used |
| data-pipeline-service | No resilience patterns used |
