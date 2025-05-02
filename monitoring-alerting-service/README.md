# Monitoring & Alerting Service

## Overview
The Monitoring & Alerting Service is responsible for tracking, analyzing, and reporting on the health and performance of the Forex Trading Platform. It provides real-time monitoring, alerting capabilities, metrics collection, and visualization dashboards to ensure system reliability and performance.

## Setup

### Prerequisites
- Python 3.10 or higher
- Poetry (dependency management)
- Docker and Docker Compose
- Prometheus (metrics database)
- Grafana (visualization)
- AlertManager (alert routing)

### Installation
1. Clone the repository
2. Navigate to the service directory:
```bash
cd monitoring-alerting-service
```
3. Install dependencies using Poetry:
```bash
poetry install
```
4. Launch the monitoring stack using Docker Compose:
```bash
docker-compose up -d
```

### Environment Variables
The following environment variables are required:

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `PORT` | Service port | 8009 |
| `PROMETHEUS_URL` | Prometheus server URL | http://prometheus:9090 |
| `ALERTMANAGER_URL` | AlertManager URL | http://alertmanager:9093 |
| `GRAFANA_URL` | Grafana URL | http://grafana:3000 |
| `API_KEY` | API key for authentication | - |
| `SLACK_WEBHOOK_URL` | Slack webhook for alerts | - |
| `EMAIL_SERVER` | SMTP server for email alerts | - |
| `EMAIL_FROM` | From address for alerts | alerts@example.com |
| `RETENTION_DAYS` | Days to keep metrics | 30 |

### Running the Service
Run the service using Poetry:
```bash
poetry run python -m monitoring_alerting_service.main
```

For development with auto-reload:
```bash
poetry run uvicorn monitoring_alerting_service.main:app --reload
```

## Monitoring Components

### Metrics Collection
The service collects metrics from various sources:

- **Application Metrics**: Performance, errors, latency from all platform services
- **Infrastructure Metrics**: CPU, memory, disk, network from hosts
- **Business Metrics**: Trading volume, order execution time, strategy performance
- **Database Metrics**: Query performance, connection pool stats
- **External API Metrics**: Availability, response time, error rates

### Dashboards
Pre-configured Grafana dashboards are available in the `dashboards/` directory:

- **Platform Overview**: High-level system health and key metrics
- **Service Performance**: Detailed metrics for each service
- **Trading Operations**: Order execution and trading metrics
- **Infrastructure**: Host-level resource utilization
- **Alerts**: Recent and active alerts

### Alerting Rules
Alert configurations are defined in `alerts/` directory:

- **Service Health**: Detect service outages and performance degradation
- **Infrastructure**: Detect resource constraints (CPU, memory, disk)
- **Trading Anomalies**: Detect unusual patterns in trading operations
- **Security**: Detect potential security issues
- **Business KPIs**: Alert on business metric thresholds

### Notification Channels
Alerts can be delivered via multiple channels:

- Slack
- Email
- SMS
- PagerDuty
- Custom webhooks

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

#### GET /api/v1/alerts
Get active alerts.

**Response:**
```json
{
  "alerts": [
    {
      "id": "high_cpu_usage",
      "severity": "warning",
      "service": "trading-gateway",
      "description": "CPU usage above 80% for 5 minutes",
      "start_time": "2025-04-29T10:15:00Z",
      "status": "firing"
    },
    {
      "id": "api_high_latency",
      "severity": "critical",
      "service": "feature-store",
      "description": "API latency above 2s for 3 minutes",
      "start_time": "2025-04-29T10:20:00Z",
      "status": "firing"
    }
  ]
}
```

#### GET /api/v1/metrics/{service}
Get metrics for a specific service.

**Parameters:**
- `service` (path): The service name
- `metric` (query): The metric name (optional)
- `start_time` (query): Start time in ISO format
- `end_time` (query): End time in ISO format
- `step` (query): Step interval (e.g., "30s", "1m", "5m")

**Response:**
```json
{
  "service": "trading-gateway",
  "metric": "http_request_duration_seconds",
  "timestamps": ["2025-04-29T10:00:00Z", "2025-04-29T10:01:00Z", "2025-04-29T10:02:00Z"],
  "values": [0.15, 0.18, 0.12]
}
```

#### POST /api/v1/alerts/silence
Create a silence for alerts.

**Request Body:**
```json
{
  "matcher": {
    "service": "trading-gateway",
    "severity": "warning"
  },
  "duration_minutes": 60,
  "created_by": "admin",
  "comment": "Investigating high CPU usage"
}
```

## Integration with Other Services
The Monitoring & Alerting Service integrates with all platform services by:

- Collecting metrics from all services
- Providing health checks for service dependencies
- Centralizing logging and error reporting
- Distributing performance metrics

## Core Monitoring Utilities
The service extracts core monitoring utilities to common-lib, including:

- Metric collection middleware
- Standardized health check implementation
- Latency tracking utilities
- Error reporting mechanisms

## Architecture
The service uses a scalable, multi-level architecture:

1. **Collectors**: Gather metrics from services and infrastructure
2. **Storage**: Time-series database (Prometheus) for metrics
3. **Processing**: Alert rules and aggregations
4. **Visualization**: Dashboards for different user needs
5. **Notification**: Multi-channel alert delivery

## Error Handling
The service implements robust error handling to ensure reliable monitoring even during platform issues.
