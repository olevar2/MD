# Environment Variables Documentation

This document provides a comprehensive list of all environment variables used across the Forex Trading Platform services. It serves as a central reference for configuring the platform.

## Common Environment Variables

These environment variables are common across multiple services:

### Service Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `SERVICE_NAME` | Name of the service | - | Yes |
| `APP_VERSION` | Version of the service | `0.1.0` | No |
| `DEBUG_MODE` | Enable debug mode | `False` | No |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | `INFO` | No |
| `HOST` | Host to bind to | `0.0.0.0` | No |
| `PORT` | Port to bind to | - | Yes |

### Database Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DB_USER` | Database user | `postgres` | Yes |
| `DB_PASSWORD` | Database password | - | Yes |
| `DB_HOST` | Database host | `localhost` | Yes |
| `DB_PORT` | Database port | `5432` | No |
| `DB_NAME` | Database name | - | Yes |
| `DB_POOL_SIZE` | Database connection pool size | `20` | No |
| `DB_SSL_REQUIRED` | Database SSL required | `False` | No |
| `DATABASE_URL` | Full database URL (overrides individual settings) | - | No |

### Redis Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `REDIS_HOST` | Redis host | `localhost` | No |
| `REDIS_PORT` | Redis port | `6379` | No |
| `REDIS_DB` | Redis database index | `0` | No |
| `REDIS_PASSWORD` | Redis password | - | No |
| `REDIS_TIMEOUT` | Redis connection timeout (seconds) | `5` | No |
| `REDIS_URL` | Full Redis URL (overrides individual settings) | - | No |

### Kafka Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka bootstrap servers | `localhost:9092` | No |
| `KAFKA_CONSUMER_GROUP_PREFIX` | Kafka consumer group prefix | - | No |
| `KAFKA_AUTO_CREATE_TOPICS` | Automatically create Kafka topics | `True` | No |
| `KAFKA_PRODUCER_ACKS` | Kafka producer acknowledgments | `all` | No |

### Monitoring Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENABLE_PROMETHEUS_METRICS` | Enable Prometheus metrics | `True` | No |
| `PROMETHEUS_METRICS_PORT` | Prometheus metrics port | - | No |
| `ENABLE_DISTRIBUTED_TRACING` | Enable distributed tracing | `False` | No |
| `JAEGER_AGENT_HOST` | Jaeger agent host | `localhost` | No |
| `JAEGER_AGENT_PORT` | Jaeger agent port | `6831` | No |

## Service-Specific Environment Variables

### Data Pipeline Service

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MAX_REQUESTS_PER_MINUTE` | Maximum requests per minute | `60` | No |
| `MAX_RETRIES` | Maximum retries | `3` | No |
| `RETRY_DELAY_SECONDS` | Retry delay (seconds) | `5` | No |
| `TIMEOUT_SECONDS` | Request timeout (seconds) | `30` | No |
| `OANDA_API_KEY` | Oanda API key | - | Yes (if using Oanda) |
| `OANDA_ACCOUNT_ID` | Oanda account ID | - | Yes (if using Oanda) |
| `USE_OBJECT_STORAGE` | Use object storage | `False` | No |
| `OBJECT_STORAGE_ENDPOINT` | Object storage endpoint | - | No |
| `OBJECT_STORAGE_KEY` | Object storage key | - | No |
| `OBJECT_STORAGE_SECRET` | Object storage secret | - | No |
| `OBJECT_STORAGE_BUCKET` | Object storage bucket | - | No |

### Feature Store Service

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FEATURE_CACHE_TTL` | Feature cache TTL (seconds) | `300` | No |
| `MAX_CACHE_SIZE` | Maximum cache size | `1000` | No |
| `ENABLE_GPU_ACCELERATION` | Enable GPU acceleration | `False` | No |
| `DATA_PIPELINE_SERVICE_URL` | Data Pipeline Service URL | - | Yes |
| `DATA_PIPELINE_SERVICE_API_KEY` | Data Pipeline Service API key | - | No |

### Analysis Engine Service

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MODEL_CACHE_TTL` | Model cache TTL (seconds) | `600` | No |
| `MAX_MODEL_CACHE_SIZE` | Maximum model cache size | `100` | No |
| `ENABLE_GPU_ACCELERATION` | Enable GPU acceleration | `False` | No |
| `FEATURE_STORE_SERVICE_URL` | Feature Store Service URL | - | Yes |
| `FEATURE_STORE_SERVICE_API_KEY` | Feature Store Service API key | - | No |

### ML Integration Service

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MODEL_STORAGE_PATH` | Model storage path | `./models` | No |
| `ENABLE_GPU_ACCELERATION` | Enable GPU acceleration | `False` | No |
| `ANALYSIS_ENGINE_SERVICE_URL` | Analysis Engine Service URL | - | Yes |
| `ANALYSIS_ENGINE_SERVICE_API_KEY` | Analysis Engine Service API key | - | No |
| `FEATURE_STORE_SERVICE_URL` | Feature Store Service URL | - | Yes |
| `FEATURE_STORE_SERVICE_API_KEY` | Feature Store Service API key | - | No |

### Trading Gateway Service

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ORDER_TIMEOUT_SECONDS` | Order timeout (seconds) | `10` | No |
| `MAX_ORDER_RETRIES` | Maximum order retries | `3` | No |
| `ENABLE_PAPER_TRADING` | Enable paper trading | `True` | No |
| `RISK_MANAGEMENT_SERVICE_URL` | Risk Management Service URL | - | Yes |
| `RISK_MANAGEMENT_SERVICE_API_KEY` | Risk Management Service API key | - | No |
| `PORTFOLIO_MANAGEMENT_SERVICE_URL` | Portfolio Management Service URL | - | Yes |
| `PORTFOLIO_MANAGEMENT_SERVICE_API_KEY` | Portfolio Management Service API key | - | No |
| `ANALYSIS_ENGINE_SERVICE_URL` | Analysis Engine Service URL | - | Yes |
| `ANALYSIS_ENGINE_SERVICE_API_KEY` | Analysis Engine Service API key | - | No |

### Portfolio Management Service

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ACCOUNT_CACHE_TTL` | Account cache TTL (seconds) | `60` | No |
| `POSITION_CACHE_TTL` | Position cache TTL (seconds) | `30` | No |
| `TRADING_GATEWAY_SERVICE_URL` | Trading Gateway Service URL | - | Yes |
| `TRADING_GATEWAY_SERVICE_API_KEY` | Trading Gateway Service API key | - | No |
| `RISK_MANAGEMENT_SERVICE_URL` | Risk Management Service URL | - | Yes |
| `RISK_MANAGEMENT_SERVICE_API_KEY` | Risk Management Service API key | - | No |

### Monitoring Alerting Service

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ALERT_CHECK_INTERVAL` | Alert check interval (seconds) | `60` | No |
| `NOTIFICATION_CHANNELS` | Notification channels (comma-separated) | `email` | No |
| `EMAIL_SMTP_SERVER` | Email SMTP server | - | Yes (if using email) |
| `EMAIL_SMTP_PORT` | Email SMTP port | `587` | No |
| `EMAIL_USERNAME` | Email username | - | Yes (if using email) |
| `EMAIL_PASSWORD` | Email password | - | Yes (if using email) |
| `EMAIL_FROM` | Email from address | - | Yes (if using email) |
| `EMAIL_TO` | Email to address | - | Yes (if using email) |

## Environment Configuration

### Development Environment

For development, you can use the following configuration:

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=postgres

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Feature Flags
ENABLE_GPU_ACCELERATION=False
ENABLE_DISTRIBUTED_COMPUTING=False
ENABLE_ADVANCED_INDICATORS=True

# Logging
LOG_LEVEL=DEBUG
```

### Production Environment

For production, you should use a more secure configuration:

```bash
# Database Configuration
DB_HOST=your-db-host
DB_PORT=5432
DB_USER=your-db-user
DB_PASSWORD=your-secure-password
DB_SSL_REQUIRED=True

# Redis Configuration
REDIS_HOST=your-redis-host
REDIS_PORT=6379
REDIS_PASSWORD=your-secure-redis-password

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=your-kafka-host:9092

# Feature Flags
ENABLE_GPU_ACCELERATION=True
ENABLE_DISTRIBUTED_COMPUTING=True
ENABLE_ADVANCED_INDICATORS=True

# Logging
LOG_LEVEL=WARNING
```

## Managing Environment Variables

### Using .env Files

Each service has a `.env.example` file that lists all the environment variables used by that service. To configure a service, copy the `.env.example` file to `.env` and update the values as needed.

```bash
cp service-name/.env.example service-name/.env
```

### Using Environment Variable Scripts

The platform includes scripts to help manage environment variables:

- `scripts/validate_env_config.py`: Validates that all required environment variables are set
- `scripts/generate_env_files.py`: Generates .env files for all services based on a configuration file

#### Validating Environment Configuration

```bash
python scripts/validate_env_config.py --service service-name --env-file service-name/.env
```

#### Generating Environment Files

```bash
python scripts/generate_env_files.py --env development --output-dir .
```
