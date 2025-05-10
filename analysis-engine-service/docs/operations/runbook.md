# Analysis Engine Operational Runbook

This runbook provides procedures for operating and maintaining the Analysis Engine service in production.

## Table of Contents

1. [Service Overview](#service-overview)
2. [Deployment](#deployment)
3. [Monitoring](#monitoring)
4. [Alerting](#alerting)
5. [Scaling](#scaling)
6. [Troubleshooting](#troubleshooting)
7. [Backup and Recovery](#backup-and-recovery)
8. [Security](#security)
9. [Maintenance](#maintenance)
10. [Disaster Recovery](#disaster-recovery)

## Service Overview

The Analysis Engine service provides technical analysis capabilities for the forex trading platform, including confluence detection, divergence analysis, and other multi-asset analysis functions.

### Key Components

- **OptimizedConfluenceDetector**: Detects confluence signals across multiple currency pairs
- **AdaptiveCacheManager**: Provides caching with adaptive TTL
- **OptimizedParallelProcessor**: Provides efficient parallel processing
- **MemoryOptimizedDataFrame**: Provides memory-efficient operations on pandas DataFrames
- **DistributedTracer**: Provides distributed tracing capabilities
- **GPUAccelerator**: Provides GPU acceleration for technical indicator calculations
- **PredictiveCacheManager**: Provides predictive caching capabilities

### Dependencies

- **Correlation Service**: Provides correlation data for currency pairs
- **Price Data Service**: Provides price data for currency pairs
- **Currency Strength Service**: Provides currency strength data

### Service Level Objectives (SLOs)

- **Availability**: 99.9% uptime
- **Latency**: 95th percentile latency < 500ms
- **Error Rate**: < 0.1% error rate

## Deployment

### Prerequisites

- Kubernetes cluster with at least 3 nodes
- Helm 3.0+
- kubectl 1.18+
- Docker registry access

### Deployment Procedure

1. **Prepare Configuration**

   ```bash
   # Create namespace
   kubectl create namespace forex-platform
   
   # Create config map
   kubectl create configmap analysis-engine-config \
     --from-file=config.yaml \
     --namespace forex-platform
   
   # Create secrets
   kubectl create secret generic analysis-engine-secrets \
     --from-literal=api-key=<API_KEY> \
     --namespace forex-platform
   ```

2. **Deploy with Helm**

   ```bash
   # Add Helm repository
   helm repo add forex-platform https://charts.forex-platform.com
   
   # Update repositories
   helm repo update
   
   # Install chart
   helm install analysis-engine forex-platform/analysis-engine \
     --namespace forex-platform \
     --values values.yaml
   ```

3. **Verify Deployment**

   ```bash
   # Check pods
   kubectl get pods -n forex-platform
   
   # Check services
   kubectl get svc -n forex-platform
   
   # Check logs
   kubectl logs -l app=analysis-engine -n forex-platform
   ```

### Configuration Options

Key configuration options in `values.yaml`:

```yaml
replicaCount: 3

resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 2
    memory: 2Gi

cache:
  ttlMinutes: 60
  maxSize: 10000

parallelProcessing:
  maxWorkers: 4

gpu:
  enabled: false
  memoryLimitMb: 1024

tracing:
  enabled: true
  samplingRate: 0.1
  endpoint: http://jaeger-collector:4317
```

## Monitoring

### Metrics

The service exposes Prometheus metrics at `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| http_requests_total | Counter | Total number of HTTP requests |
| http_request_duration_seconds | Histogram | HTTP request duration |
| confluence_detection_total | Counter | Total number of confluence detections |
| confluence_detection_duration_seconds | Histogram | Confluence detection duration |
| divergence_analysis_total | Counter | Total number of divergence analyses |
| divergence_analysis_duration_seconds | Histogram | Divergence analysis duration |
| cache_size | Gauge | Cache size |
| cache_hit_rate | Gauge | Cache hit rate |
| memory_usage_bytes | Gauge | Memory usage |
| cpu_usage_seconds_total | Counter | CPU usage |

### Dashboards

Grafana dashboards are available at:

- **Analysis Engine Overview**: General service metrics
- **Confluence & Divergence**: Detailed metrics for confluence and divergence detection
- **Cache & Resources**: Metrics for cache and resource usage

### Logs

Logs are sent to stdout/stderr and collected by Fluentd:

```bash
# View logs
kubectl logs -l app=analysis-engine -n forex-platform

# View logs with Kibana
# Access Kibana at http://kibana.forex-platform.local
```

### Traces

Distributed traces are collected by Jaeger:

```bash
# Access Jaeger UI
# http://jaeger.forex-platform.local
```

## Alerting

### Alert Rules

| Alert | Condition | Severity | Description |
|-------|-----------|----------|-------------|
| HighRequestLatency | p95 latency > 1s | Warning | High request latency |
| HighErrorRate | Error rate > 5% | Warning | High error rate |
| InstanceDown | Instance down > 1m | Critical | Instance down |
| HighMemoryUsage | Memory > 1.5GB | Warning | High memory usage |
| HighCPUUsage | CPU > 80% | Warning | High CPU usage |
| LowCacheHitRate | Hit rate < 50% | Warning | Low cache hit rate |

### Alert Notifications

Alerts are sent to:

- **Slack**: #forex-platform-alerts
- **Email**: alerts@forex-platform.com
- **PagerDuty**: Forex Platform team

## Scaling

### Horizontal Scaling

The service uses Horizontal Pod Autoscaler (HPA) for automatic scaling:

```bash
# View HPA status
kubectl get hpa -n forex-platform
```

HPA configuration:

- **Min Replicas**: 2
- **Max Replicas**: 10
- **CPU Target**: 70%
- **Memory Target**: 80%

### Vertical Scaling

To adjust resource limits:

```bash
# Update resource limits
kubectl set resources deployment analysis-engine \
  --requests=cpu=1,memory=1Gi \
  --limits=cpu=4,memory=4Gi \
  -n forex-platform
```

### Manual Scaling

For immediate scaling:

```bash
# Scale up
kubectl scale deployment analysis-engine --replicas=5 -n forex-platform

# Scale down
kubectl scale deployment analysis-engine --replicas=2 -n forex-platform
```

## Troubleshooting

### Common Issues

#### High Memory Usage

**Symptoms**:
- Memory usage alerts
- OOMKilled pods

**Resolution**:
1. Check cache size:
   ```bash
   curl http://analysis-engine/metrics | grep cache_size
   ```
2. Reduce cache size in config map:
   ```yaml
   cache:
     maxSize: 5000
   ```
3. Restart pods:
   ```bash
   kubectl rollout restart deployment analysis-engine -n forex-platform
   ```

#### High CPU Usage

**Symptoms**:
- CPU usage alerts
- Slow response times

**Resolution**:
1. Check parallel workers:
   ```bash
   curl http://analysis-engine/metrics | grep parallel_workers
   ```
2. Reduce max workers in config map:
   ```yaml
   parallelProcessing:
     maxWorkers: 2
   ```
3. Restart pods:
   ```bash
   kubectl rollout restart deployment analysis-engine -n forex-platform
   ```

#### Slow Response Times

**Symptoms**:
- High latency alerts
- Client timeouts

**Resolution**:
1. Check cache hit rate:
   ```bash
   curl http://analysis-engine/metrics | grep cache_hit_rate
   ```
2. If hit rate is low, check cache TTL:
   ```yaml
   cache:
     ttlMinutes: 120
   ```
3. Check for slow database queries in logs:
   ```bash
   kubectl logs -l app=analysis-engine -n forex-platform | grep "slow query"
   ```

### Debugging

#### View Pod Details

```bash
# Get pod details
kubectl describe pod <pod-name> -n forex-platform
```

#### Check Logs

```bash
# View logs
kubectl logs <pod-name> -n forex-platform

# View logs with tail
kubectl logs -f <pod-name> -n forex-platform
```

#### Check Events

```bash
# View events
kubectl get events -n forex-platform
```

#### Exec into Pod

```bash
# Exec into pod
kubectl exec -it <pod-name> -n forex-platform -- /bin/bash
```

## Backup and Recovery

### Cache Backup

The cache is ephemeral and does not require backup.

### Configuration Backup

Backup configuration:

```bash
# Backup config map
kubectl get configmap analysis-engine-config -n forex-platform -o yaml > analysis-engine-config.yaml

# Backup secrets
kubectl get secret analysis-engine-secrets -n forex-platform -o yaml > analysis-engine-secrets.yaml
```

### Recovery Procedure

Restore configuration:

```bash
# Restore config map
kubectl apply -f analysis-engine-config.yaml

# Restore secrets
kubectl apply -f analysis-engine-secrets.yaml

# Restart pods
kubectl rollout restart deployment analysis-engine -n forex-platform
```

## Security

### Authentication

The service uses API keys for authentication:

```bash
# Create API key
kubectl create secret generic analysis-engine-api-key \
  --from-literal=api-key=$(openssl rand -hex 32) \
  --namespace forex-platform
```

### Authorization

The service uses RBAC for authorization:

```bash
# View RBAC roles
kubectl get roles -n forex-platform
```

### Network Security

The service is protected by network policies:

```bash
# View network policies
kubectl get networkpolicies -n forex-platform
```

### Secrets Management

Secrets are managed with Kubernetes secrets:

```bash
# View secrets
kubectl get secrets -n forex-platform
```

## Maintenance

### Updating the Service

```bash
# Update Helm chart
helm upgrade analysis-engine forex-platform/analysis-engine \
  --namespace forex-platform \
  --values values.yaml
```

### Rolling Restart

```bash
# Perform rolling restart
kubectl rollout restart deployment analysis-engine -n forex-platform
```

### Scheduled Maintenance

- **Cache Cleanup**: Automatic, every 5 minutes
- **Log Rotation**: Automatic, daily
- **Metrics Retention**: 15 days

## Disaster Recovery

### Failover Procedure

1. **Identify Failure**:
   - Monitor alerts and dashboards
   - Confirm service unavailability

2. **Activate Standby Region**:
   ```bash
   # Switch to standby region
   kubectl config use-context standby-region
   
   # Verify standby region
   kubectl get nodes
   ```

3. **Deploy Service**:
   ```bash
   # Deploy service in standby region
   helm install analysis-engine forex-platform/analysis-engine \
     --namespace forex-platform \
     --values values-dr.yaml
   ```

4. **Update DNS**:
   ```bash
   # Update DNS to point to standby region
   kubectl apply -f dns-update.yaml
   ```

5. **Verify Service**:
   ```bash
   # Verify service is running
   kubectl get pods -n forex-platform
   
   # Test service
   curl https://api.forex-platform.com/analysis-engine/health
   ```

### Recovery Time Objective (RTO)

- **Target RTO**: 15 minutes
- **Actual RTO**: Typically 5-10 minutes

### Recovery Point Objective (RPO)

- **Target RPO**: 0 minutes (no data loss)
- **Actual RPO**: 0 minutes (stateless service)
