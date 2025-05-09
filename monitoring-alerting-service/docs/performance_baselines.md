# Performance Baselines

This document defines the performance baselines for all services in the forex trading platform. These baselines are used to set appropriate thresholds for alerts and to track performance improvements over time.

## Test Scenarios

### Scenario: normal_load

- **Description**: Normal trading hours with moderate market activity
- **Concurrent Users**: 10
- **Duration**: 60 seconds
- **Requests Per Second**: 10

### Scenario: high_load

- **Description**: Market open with high activity
- **Concurrent Users**: 50
- **Duration**: 60 seconds
- **Requests Per Second**: 50

### Scenario: peak_load

- **Description**: News event with very high activity
- **Concurrent Users**: 100
- **Duration**: 30 seconds
- **Requests Per Second**: 100

## Service Baselines

### trading-gateway-service

#### API Performance

| Endpoint | Scenario | p50 Latency | p95 Latency | p99 Latency | Throughput | Error Rate |
|----------|----------|------------|------------|------------|------------|------------|
| list_orders | normal_load | 100.00ms | 200.00ms | 500.00ms | 10.00 req/s | 1.00% |
| list_orders | high_load | 100.00ms | 200.00ms | 500.00ms | 50.00 req/s | 1.00% |
| list_orders | peak_load | 100.00ms | 200.00ms | 500.00ms | 100.00 req/s | 1.00% |
| create_order | normal_load | 100.00ms | 200.00ms | 500.00ms | 10.00 req/s | 1.00% |
| create_order | high_load | 100.00ms | 200.00ms | 500.00ms | 50.00 req/s | 1.00% |
| create_order | peak_load | 100.00ms | 200.00ms | 500.00ms | 100.00 req/s | 1.00% |
| list_positions | normal_load | 100.00ms | 200.00ms | 500.00ms | 10.00 req/s | 1.00% |
| list_positions | high_load | 100.00ms | 200.00ms | 500.00ms | 50.00 req/s | 1.00% |
| list_positions | peak_load | 100.00ms | 200.00ms | 500.00ms | 100.00 req/s | 1.00% |

#### Resource Usage

| Metric | Baseline | Warning Threshold | Critical Threshold |
|--------|----------|-------------------|--------------------|
| cpu_usage | 30% | 70% | 90% |
| memory_usage | 40% | 80% | 90% |

#### Business Metrics

| Metric | Baseline | Warning Threshold | Critical Threshold |
|--------|----------|-------------------|--------------------|
| order_execution_time | 0.8 | 0.6 | 0.5 |
| slippage_bps | 0.8 | 0.6 | 0.5 |
| fill_rate | 0.8 | 0.6 | 0.5 |

### analysis-engine-service

#### API Performance

| Endpoint | Scenario | p50 Latency | p95 Latency | p99 Latency | Throughput | Error Rate |
|----------|----------|------------|------------|------------|------------|------------|
| market_analysis | normal_load | 100.00ms | 200.00ms | 500.00ms | 10.00 req/s | 1.00% |
| market_analysis | high_load | 100.00ms | 200.00ms | 500.00ms | 50.00 req/s | 1.00% |
| market_analysis | peak_load | 100.00ms | 200.00ms | 500.00ms | 100.00 req/s | 1.00% |
| pattern_detection | normal_load | 100.00ms | 200.00ms | 500.00ms | 10.00 req/s | 1.00% |
| pattern_detection | high_load | 100.00ms | 200.00ms | 500.00ms | 50.00 req/s | 1.00% |
| pattern_detection | peak_load | 100.00ms | 200.00ms | 500.00ms | 100.00 req/s | 1.00% |
| signal_generation | normal_load | 100.00ms | 200.00ms | 500.00ms | 10.00 req/s | 1.00% |
| signal_generation | high_load | 100.00ms | 200.00ms | 500.00ms | 50.00 req/s | 1.00% |
| signal_generation | peak_load | 100.00ms | 200.00ms | 500.00ms | 100.00 req/s | 1.00% |

#### Resource Usage

| Metric | Baseline | Warning Threshold | Critical Threshold |
|--------|----------|-------------------|--------------------|
| cpu_usage | 30% | 70% | 90% |
| memory_usage | 40% | 80% | 90% |

#### Business Metrics

| Metric | Baseline | Warning Threshold | Critical Threshold |
|--------|----------|-------------------|--------------------|
| pattern_recognition_accuracy | 0.8 | 0.6 | 0.5 |
| signal_quality_score | 0.8 | 0.6 | 0.5 |
| market_regime_detection_confidence | 0.8 | 0.6 | 0.5 |

### feature-store-service

#### API Performance

| Endpoint | Scenario | p50 Latency | p95 Latency | p99 Latency | Throughput | Error Rate |
|----------|----------|------------|------------|------------|------------|------------|
| list_features | normal_load | 100.00ms | 200.00ms | 500.00ms | 10.00 req/s | 1.00% |
| list_features | high_load | 100.00ms | 200.00ms | 500.00ms | 50.00 req/s | 1.00% |
| list_features | peak_load | 100.00ms | 200.00ms | 500.00ms | 100.00 req/s | 1.00% |
| calculate_features | normal_load | 100.00ms | 200.00ms | 500.00ms | 10.00 req/s | 1.00% |
| calculate_features | high_load | 100.00ms | 200.00ms | 500.00ms | 50.00 req/s | 1.00% |
| calculate_features | peak_load | 100.00ms | 200.00ms | 500.00ms | 100.00 req/s | 1.00% |
| batch_calculate | normal_load | 100.00ms | 200.00ms | 500.00ms | 10.00 req/s | 1.00% |
| batch_calculate | high_load | 100.00ms | 200.00ms | 500.00ms | 50.00 req/s | 1.00% |
| batch_calculate | peak_load | 100.00ms | 200.00ms | 500.00ms | 100.00 req/s | 1.00% |

#### Resource Usage

| Metric | Baseline | Warning Threshold | Critical Threshold |
|--------|----------|-------------------|--------------------|
| cpu_usage | 30% | 70% | 90% |
| memory_usage | 40% | 80% | 90% |

#### Business Metrics

| Metric | Baseline | Warning Threshold | Critical Threshold |
|--------|----------|-------------------|--------------------|
| feature_calculation_time | 0.8 | 0.6 | 0.5 |
| cache_hit_rate | 0.8 | 0.6 | 0.5 |
| data_freshness | 0.8 | 0.6 | 0.5 |

### ml-integration-service

#### API Performance

| Endpoint | Scenario | p50 Latency | p95 Latency | p99 Latency | Throughput | Error Rate |
|----------|----------|------------|------------|------------|------------|------------|
| list_models | normal_load | 100.00ms | 200.00ms | 500.00ms | 10.00 req/s | 1.00% |
| list_models | high_load | 100.00ms | 200.00ms | 500.00ms | 50.00 req/s | 1.00% |
| list_models | peak_load | 100.00ms | 200.00ms | 500.00ms | 100.00 req/s | 1.00% |
| model_prediction | normal_load | 100.00ms | 200.00ms | 500.00ms | 10.00 req/s | 1.00% |
| model_prediction | high_load | 100.00ms | 200.00ms | 500.00ms | 50.00 req/s | 1.00% |
| model_prediction | peak_load | 100.00ms | 200.00ms | 500.00ms | 100.00 req/s | 1.00% |
| batch_prediction | normal_load | 100.00ms | 200.00ms | 500.00ms | 10.00 req/s | 1.00% |
| batch_prediction | high_load | 100.00ms | 200.00ms | 500.00ms | 50.00 req/s | 1.00% |
| batch_prediction | peak_load | 100.00ms | 200.00ms | 500.00ms | 100.00 req/s | 1.00% |

#### Resource Usage

| Metric | Baseline | Warning Threshold | Critical Threshold |
|--------|----------|-------------------|--------------------|
| cpu_usage | 30% | 70% | 90% |
| memory_usage | 40% | 80% | 90% |

#### Business Metrics

| Metric | Baseline | Warning Threshold | Critical Threshold |
|--------|----------|-------------------|--------------------|
| model_inference_time | 0.8 | 0.6 | 0.5 |
| prediction_accuracy | 0.8 | 0.6 | 0.5 |
| model_confidence | 0.8 | 0.6 | 0.5 |

### strategy-execution-engine

#### API Performance

| Endpoint | Scenario | p50 Latency | p95 Latency | p99 Latency | Throughput | Error Rate |
|----------|----------|------------|------------|------------|------------|------------|
| list_strategies | normal_load | 100.00ms | 200.00ms | 500.00ms | 10.00 req/s | 1.00% |
| list_strategies | high_load | 100.00ms | 200.00ms | 500.00ms | 50.00 req/s | 1.00% |
| list_strategies | peak_load | 100.00ms | 200.00ms | 500.00ms | 100.00 req/s | 1.00% |
| execute_strategy | normal_load | 100.00ms | 200.00ms | 500.00ms | 10.00 req/s | 1.00% |
| execute_strategy | high_load | 100.00ms | 200.00ms | 500.00ms | 50.00 req/s | 1.00% |
| execute_strategy | peak_load | 100.00ms | 200.00ms | 500.00ms | 100.00 req/s | 1.00% |
| backtest_strategy | normal_load | 100.00ms | 200.00ms | 500.00ms | 10.00 req/s | 1.00% |
| backtest_strategy | high_load | 100.00ms | 200.00ms | 500.00ms | 50.00 req/s | 1.00% |
| backtest_strategy | peak_load | 100.00ms | 200.00ms | 500.00ms | 100.00 req/s | 1.00% |

#### Resource Usage

| Metric | Baseline | Warning Threshold | Critical Threshold |
|--------|----------|-------------------|--------------------|
| cpu_usage | 30% | 70% | 90% |
| memory_usage | 40% | 80% | 90% |

#### Business Metrics

| Metric | Baseline | Warning Threshold | Critical Threshold |
|--------|----------|-------------------|--------------------|
| strategy_execution_time | 0.8 | 0.6 | 0.5 |
| strategy_win_rate | 0.8 | 0.6 | 0.5 |
| strategy_sharpe_ratio | 0.8 | 0.6 | 0.5 |

### data-pipeline-service

#### API Performance

| Endpoint | Scenario | p50 Latency | p95 Latency | p99 Latency | Throughput | Error Rate |
|----------|----------|------------|------------|------------|------------|------------|
| get_market_data | normal_load | 100.00ms | 200.00ms | 500.00ms | 10.00 req/s | 1.00% |
| get_market_data | high_load | 100.00ms | 200.00ms | 500.00ms | 50.00 req/s | 1.00% |
| get_market_data | peak_load | 100.00ms | 200.00ms | 500.00ms | 100.00 req/s | 1.00% |
| process_data | normal_load | 100.00ms | 200.00ms | 500.00ms | 10.00 req/s | 1.00% |
| process_data | high_load | 100.00ms | 200.00ms | 500.00ms | 50.00 req/s | 1.00% |
| process_data | peak_load | 100.00ms | 200.00ms | 500.00ms | 100.00 req/s | 1.00% |
| pipeline_status | normal_load | 100.00ms | 200.00ms | 500.00ms | 10.00 req/s | 1.00% |
| pipeline_status | high_load | 100.00ms | 200.00ms | 500.00ms | 50.00 req/s | 1.00% |
| pipeline_status | peak_load | 100.00ms | 200.00ms | 500.00ms | 100.00 req/s | 1.00% |

#### Resource Usage

| Metric | Baseline | Warning Threshold | Critical Threshold |
|--------|----------|-------------------|--------------------|
| cpu_usage | 30% | 70% | 90% |
| memory_usage | 40% | 80% | 90% |

#### Business Metrics

| Metric | Baseline | Warning Threshold | Critical Threshold |
|--------|----------|-------------------|--------------------|
| data_processing_time | 0.8 | 0.6 | 0.5 |
| data_quality_score | 0.8 | 0.6 | 0.5 |
| pipeline_throughput | 0.8 | 0.6 | 0.5 |

