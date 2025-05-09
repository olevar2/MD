# Prometheus Exporters for Forex Trading Platform

This directory contains Docker Compose configuration for various Prometheus exporters used to monitor the Forex Trading Platform infrastructure.

## Included Exporters

### Database Monitoring
- **PostgreSQL Exporter**: Collects metrics from PostgreSQL/TimescaleDB databases
  - Port: 9187
  - Metrics: Query performance, connections, locks, replication status, etc.

### Message Queue Monitoring
- **Kafka Exporter**: Collects metrics from Kafka message brokers
  - Port: 9308
  - Metrics: Topic lag, consumer group status, message rates, etc.

### Cache Monitoring
- **Redis Exporter**: Collects metrics from Redis cache
  - Port: 9121
  - Metrics: Memory usage, hit/miss rates, connected clients, etc.

### System Monitoring
- **Node Exporter**: Collects host-level metrics
  - Port: 9100
  - Metrics: CPU, memory, disk, network usage, etc.

- **cAdvisor**: Collects container-level metrics
  - Port: 8080
  - Metrics: Container CPU, memory, network usage, etc.

## Setup Instructions

1. Ensure the monitoring network exists:
   ```bash
   docker network create monitoring-network
   ```

2. Start the exporters:
   ```bash
   docker-compose up -d
   ```

3. Verify the exporters are running:
   ```bash
   docker-compose ps
   ```

4. Test metrics endpoints:
   - PostgreSQL: http://localhost:9187/metrics
   - Kafka: http://localhost:9308/metrics
   - Redis: http://localhost:9121/metrics
   - Node: http://localhost:9100/metrics
   - cAdvisor: http://localhost:8080/metrics

## Configuration

### PostgreSQL Exporter
- Configure connection string in `DATA_SOURCE_NAME` environment variable
- Adjust collectors with command-line flags

### Kafka Exporter
- Configure Kafka server with `--kafka.server` flag
- Filter topics and consumer groups with `--topic.filter` and `--group.filter` flags

### Redis Exporter
- Configure Redis connection with `REDIS_ADDR` environment variable
- Set namespace with `--namespace` flag

### Node Exporter
- Mount host filesystems to collect system metrics
- Configure collectors with command-line flags

### cAdvisor
- Mount Docker socket and system directories to collect container metrics

## Integration with Prometheus

These exporters are automatically discovered by Prometheus using the configuration in `prometheus.yml`. Make sure the service names match the targets defined in the Prometheus configuration.

## Troubleshooting

If an exporter is not working:

1. Check if the container is running:
   ```bash
   docker ps | grep exporter
   ```

2. Check container logs:
   ```bash
   docker logs <container-name>
   ```

3. Verify the metrics endpoint is accessible:
   ```bash
   curl http://localhost:<port>/metrics
   ```

4. Check Prometheus targets in the Prometheus UI (http://localhost:9090/targets)
