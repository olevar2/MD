# Analysis Engine Service Monitoring

This directory contains monitoring configurations and dashboards for the Analysis Engine Service.

## Overview

The Analysis Engine Service includes comprehensive monitoring capabilities:

1. **Metrics Collection**: Prometheus metrics for all key operations
2. **Structured Logging**: Logs with correlation IDs and context data
3. **Health Checks**: Detailed component and dependency health checks
4. **Dashboards**: Grafana dashboards for visualizing service performance

## Metrics

The service exposes Prometheus metrics at the `/metrics` endpoint. Key metrics include:

- **Analysis Operation Metrics**:
  - `analysis_engine_requests_total`: Total number of analysis requests
  - `analysis_engine_errors_total`: Total number of analysis errors
  - `analysis_engine_duration_seconds`: Duration of analysis operations

- **Resource Usage Metrics**:
  - `analysis_engine_resource_usage`: Resource usage percentage (CPU, memory, disk)

- **Cache Metrics**:
  - `analysis_engine_cache_operations_total`: Total number of cache operations
  - `analysis_engine_cache_hits_total`: Total number of cache hits
  - `analysis_engine_cache_misses_total`: Total number of cache misses

- **Dependency Health Metrics**:
  - `analysis_engine_dependency_health`: Health status of dependencies
  - `analysis_engine_dependency_latency_seconds`: Latency of dependency operations

- **API Metrics**:
  - `analysis_engine_api_duration_seconds`: Duration of API requests
  - `analysis_engine_api_requests_total`: Total number of API requests

## Health Checks

The service provides health check endpoints:

- `/api/health/live`: Liveness probe (simple check that service is running)
- `/api/health/ready`: Readiness probe (check if service is ready to receive traffic)
- `/api/health`: Detailed health check with component and dependency status

## Dashboards

The `grafana-dashboard.json` file contains a Grafana dashboard configuration that visualizes the service metrics. Import this file into your Grafana instance to use the dashboard.

## Alerts

The `alerts.yml` file contains Prometheus alerting rules for the service. These rules define alerts for high error rates, high latency, unhealthy dependencies, and resource usage.

## Prometheus Configuration

The `prometheus.yml` file contains a sample Prometheus configuration for scraping metrics from the service.

## Setup

1. **Prometheus**:
   ```bash
   docker run -d --name prometheus -p 9090:9090 -v /path/to/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
   ```

2. **Grafana**:
   ```bash
   docker run -d --name grafana -p 3000:3000 grafana/grafana
   ```

3. Import the dashboard into Grafana:
   - Go to Dashboards > Import
   - Upload the `grafana-dashboard.json` file or paste its contents

4. Configure Prometheus as a data source in Grafana:
   - Go to Configuration > Data Sources
   - Add Prometheus data source with URL `http://prometheus:9090`
