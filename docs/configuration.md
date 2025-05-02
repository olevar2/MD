# Configuration Management

## Overview

The Analysis Engine Service uses a centralized configuration management system based on Pydantic's `BaseSettings`. This document describes how to use and extend the configuration system.

## Configuration Structure

The configuration is organized in a hierarchical structure:

1. **Common Base Settings**: Defined in `common_lib.config.AppSettings`
2. **Service-Specific Settings**: Defined in `analysis_engine.config.settings.AnalysisEngineSettings`
3. **Backward Compatibility Layers**:
   - `analysis_engine.core.config` (deprecated)
   - `config.config` (deprecated)

## Using Configuration

### Recommended Approach

Import the settings directly from the centralized module:

```python
from analysis_engine.config import get_settings

# Get the settings instance
settings = get_settings()

# Access settings
host = settings.HOST
port = settings.PORT
```

### Helper Functions

For convenience, several helper functions are provided:

```python
from analysis_engine.config import get_db_settings, get_api_settings

# Get database settings
db_settings = get_db_settings()
database_url = db_settings["database_url"]

# Get API settings
api_settings = get_api_settings()
host = api_settings["host"]
port = api_settings["port"]
```

### Advanced Usage Examples

#### 1. Using Feature Flags

Feature flags allow you to enable or disable specific features without changing code:

```python
from analysis_engine.config import get_settings

settings = get_settings()

# Check if a feature is enabled
if settings.FEATURE_MULTI_TIMEFRAME_ANALYSIS:
    # Initialize multi-timeframe analyzer
    analyzer = MultiTimeframeAnalyzer()
    # Use the analyzer
    results = analyzer.analyze(data)
else:
    # Use alternative approach or skip
    logger.info("Multi-timeframe analysis is disabled")
```

#### 2. Configuring External Service Connections

Configure connections to external services with proper error handling:

```python
from analysis_engine.config import get_settings
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

settings = get_settings()

# Create a client with configured timeout and retry settings
@retry(
    stop=stop_after_attempt(settings.SERVICE_MAX_RETRIES),
    wait=wait_exponential(multiplier=settings.SERVICE_RETRY_BACKOFF)
)
async def fetch_market_data(symbol: str, timeframe: str):
    async with httpx.AsyncClient(timeout=settings.SERVICE_TIMEOUT_SECONDS) as client:
        response = await client.get(
            f"{settings.MARKET_DATA_SERVICE_URL}/data/{symbol}/{timeframe}"
        )
        response.raise_for_status()
        return response.json()
```

#### 3. Configuring Database Connection Pool

Configure database connection pool with settings:

```python
from analysis_engine.config import get_settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

settings = get_settings()

# Create engine with connection pool settings
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_recycle=settings.DB_POOL_RECYCLE
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

#### 4. Configuring Logging

Configure logging with settings:

```python
from analysis_engine.config import get_settings
import logging
from logging.handlers import RotatingFileHandler
import sys

settings = get_settings()

# Configure root logger
logger = logging.getLogger()
logger.setLevel(settings.LOG_LEVEL)

# Configure formatter
formatter = logging.Formatter(settings.LOG_FORMAT)

# Always add console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Add file handler if configured
if settings.LOG_FILE:
    if settings.LOG_ROTATION:
        file_handler = RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=settings.LOG_MAX_SIZE,
            backupCount=settings.LOG_BACKUP_COUNT
        )
    else:
        file_handler = logging.FileHandler(settings.LOG_FILE)

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
```

## Environment Variables

The configuration system loads values from environment variables. Here are the key environment variables organized by category:

### Service Metadata

| Environment Variable | Description | Default Value | Validation |
|---------------------|-------------|---------------|------------|
| `SERVICE_NAME` | Name of the service | `"analysis-engine-service"` | - |
| `DEBUG_MODE` | Enable debug mode | `False` | Boolean |
| `LOG_LEVEL` | Logging level | `"INFO"` | One of: DEBUG, INFO, WARNING, ERROR, CRITICAL |

### API Settings

| Environment Variable | Description | Default Value | Validation |
|---------------------|-------------|---------------|------------|
| `API_VERSION` | API version | `"v1"` | Format: v1, v1.0, etc. |
| `API_PREFIX` | API endpoint prefix | `"/api/v1"` | Must start with / |
| `HOST` | Host to bind the API server | `"0.0.0.0"` | Valid IP address or hostname |
| `PORT` | Port to bind the API server | `8000` | Range: 1024-65535 |

### Security Settings

| Environment Variable | Description | Default Value | Validation |
|---------------------|-------------|---------------|------------|
| `JWT_SECRET` | Secret key for JWT tokens | `"your-secret-key"` | Min length: 16 chars |
| `JWT_ALGORITHM` | Algorithm for JWT tokens | `"HS256"` | One of: HS256, HS384, HS512, RS256, etc. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT token expiration time in minutes | `30` | Range: 5-1440 |
| `CORS_ORIGINS` | Allowed CORS origins | `["*"]` | List of URLs or "*" |

### Database Settings

| Environment Variable | Description | Default Value | Validation |
|---------------------|-------------|---------------|------------|
| `DATABASE_URL` | Direct database URL | `None` | Valid SQLAlchemy URL |
| `DB_USER` | Database user | `None` | Required if DATABASE_URL not provided |
| `DB_PASSWORD` | Database password | `None` | Required if DATABASE_URL not provided |
| `DB_HOST` | Database host | `None` | Required if DATABASE_URL not provided |
| `DB_PORT` | Database port | `5432` | Range: 1-65535 |
| `DB_NAME` | Database name | `None` | Required if DATABASE_URL not provided |
| `DB_POOL_SIZE` | Database connection pool size | `5` | Range: 1-100 |
| `DB_MAX_OVERFLOW` | Maximum overflow connections | `10` | Range: 0-100 |
| `DB_POOL_TIMEOUT` | Connection pool timeout in seconds | `30` | Range: 1-300 |
| `DB_POOL_RECYCLE` | Connection recycle time in seconds | `1800` | Min: 1 |

### Redis Settings

| Environment Variable | Description | Default Value | Validation |
|---------------------|-------------|---------------|------------|
| `REDIS_HOST` | Redis host | `"localhost"` | Valid hostname or IP |
| `REDIS_PORT` | Redis port | `6379` | Range: 1-65535 |
| `REDIS_DB` | Redis database index | `0` | Range: 0-15 |
| `REDIS_PASSWORD` | Redis password | `None` | - |
| `REDIS_TIMEOUT` | Redis connection timeout in seconds | `10` | Range: 1-60 |
| `REDIS_SSL` | Use SSL for Redis connection | `False` | Boolean |
| `REDIS_POOL_SIZE` | Redis connection pool size | `10` | Range: 1-100 |

### External Services

| Environment Variable | Description | Default Value | Validation |
|---------------------|-------------|---------------|------------|
| `MARKET_DATA_SERVICE_URL` | URL for the Market Data Service | `"http://market-data-service:8000/api/v1"` | Valid URL |
| `NOTIFICATION_SERVICE_URL` | URL for the Notification Service | `"http://notification-service:8000/api/v1"` | Valid URL |
| `ML_INTEGRATION_SERVICE_URL` | URL for the ML Integration Service | `"http://ml-integration-service:8000/api/v1"` | Valid URL |
| `FEATURE_STORE_SERVICE_URL` | URL for the Feature Store Service | `"http://feature-store-service:8000/api/v1"` | Valid URL |
| `SERVICE_TIMEOUT_SECONDS` | Default timeout for external service requests | `30` | Range: 1-300 |
| `SERVICE_MAX_RETRIES` | Maximum number of retries for external service requests | `3` | Range: 0-10 |
| `SERVICE_RETRY_BACKOFF` | Backoff factor for retries in seconds | `0.5` | Range: 0.1-60.0 |

### Analysis Settings

| Environment Variable | Description | Default Value | Validation |
|---------------------|-------------|---------------|------------|
| `ANALYSIS_TIMEOUT` | Timeout for analysis operations in seconds | `30` | Range: 5-300 |
| `MAX_CONCURRENT_ANALYSES` | Maximum number of concurrent analyses | `10` | Range: 1-100 |
| `DEFAULT_TIMEFRAMES` | Default timeframes for analysis | `["M15", "H1", "H4", "D1"]` | Valid timeframes |
| `DEFAULT_SYMBOLS` | Default symbols for analysis | `["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]` | Valid symbols |
| `ENABLE_ADVANCED_ANALYSIS` | Enable advanced analysis features | `True` | Boolean |
| `ENABLE_ML_INTEGRATION` | Enable machine learning integration | `True` | Boolean |
| `ANALYSIS_CACHE_TTL` | Time-to-live for analysis cache in seconds | `3600` | Range: 0-86400 |

### Rate Limiting

| Environment Variable | Description | Default Value | Validation |
|---------------------|-------------|---------------|------------|
| `RATE_LIMIT_REQUESTS` | Standard rate limit requests per minute | `100` | Range: 1-10000 |
| `RATE_LIMIT_PERIOD` | Rate limit period in seconds | `60` | Range: 1-3600 |
| `RATE_LIMIT_PREMIUM` | Premium rate limit requests per minute | `500` | Range: 1-50000 |
| `RATE_LIMIT_BURST` | Burst capacity for rate limiting | `20` | Range: 1-1000 |

### Logging

| Environment Variable | Description | Default Value | Validation |
|---------------------|-------------|---------------|------------|
| `LOG_FORMAT` | Logging format string | `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"` | Must include %(levelname)s and %(message)s |
| `LOG_FILE` | Log file path | `None` | - |
| `LOG_ROTATION` | Enable log rotation | `True` | Boolean |
| `LOG_MAX_SIZE` | Maximum log file size in bytes before rotation | `10485760` (10 MB) | Range: 1MB-100MB |
| `LOG_BACKUP_COUNT` | Number of backup log files to keep | `5` | Range: 0-100 |

### Performance Monitoring

| Environment Variable | Description | Default Value | Validation |
|---------------------|-------------|---------------|------------|
| `ENABLE_METRICS` | Enable performance metrics collection | `True` | Boolean |
| `METRICS_PORT` | Port for metrics server | `9090` | Range: 1024-65535 |
| `ENABLE_TRACING` | Enable distributed tracing | `False` | Boolean |
| `TRACING_SAMPLE_RATE` | Sampling rate for distributed tracing | `0.1` | Range: 0.0-1.0 |

### WebSocket Settings

| Environment Variable | Description | Default Value | Validation |
|---------------------|-------------|---------------|------------|
| `WS_HEARTBEAT_INTERVAL` | WebSocket heartbeat interval in seconds | `30` | Range: 5-300 |
| `WS_MAX_CONNECTIONS` | Maximum number of WebSocket connections | `1000` | Range: 10-10000 |
| `WS_CLOSE_TIMEOUT` | Timeout for WebSocket close handshake in seconds | `10` | Range: 1-60 |

### Feature Flags

| Environment Variable | Description | Default Value | Validation |
|---------------------|-------------|---------------|------------|
| `FEATURE_MULTI_TIMEFRAME_ANALYSIS` | Enable multi-timeframe analysis | `True` | Boolean |
| `FEATURE_MARKET_REGIME_DETECTION` | Enable market regime detection | `True` | Boolean |
| `FEATURE_SENTIMENT_ANALYSIS` | Enable sentiment analysis | `True` | Boolean |
| `FEATURE_CORRELATION_ANALYSIS` | Enable correlation analysis | `True` | Boolean |

## Configuration Manager

For dynamic configuration management, a `ConfigurationManager` class is provided:

```python
from analysis_engine.config import ConfigurationManager

# Create a configuration manager
config_manager = ConfigurationManager()

# Get a configuration value
value = config_manager.get("HOST")

# Set a configuration value
config_manager.set("HOST", "new_host")

# Reload configuration from environment
config_manager.reload()
```

### Advanced Configuration Manager Usage

The Configuration Manager provides a caching layer and dynamic configuration capabilities:

```python
from analysis_engine.config import ConfigurationManager
import logging

logger = logging.getLogger(__name__)

# Create a configuration manager
config_manager = ConfigurationManager()

# Get configuration with fallback
db_timeout = config_manager.get("DB_POOL_TIMEOUT", 30)

# Use configuration in a context
with db_connection_pool(timeout=db_timeout) as conn:
    # Do something with the connection
    pass

# Dynamic configuration updates
def update_rate_limits(new_limit: int):
    """Update rate limits dynamically."""
    try:
        # Validate the new limit
        if new_limit < 1 or new_limit > 10000:
            raise ValueError(f"Invalid rate limit: {new_limit}")

        # Update the configuration
        old_limit = config_manager.get("RATE_LIMIT_REQUESTS")
        config_manager.set("RATE_LIMIT_REQUESTS", new_limit)
        logger.info(f"Rate limit updated from {old_limit} to {new_limit}")

        # Return success
        return True
    except Exception as e:
        logger.error(f"Failed to update rate limit: {e}")
        return False
```

## Extending Configuration

To add new configuration parameters, update the `AnalysisEngineSettings` class in `analysis_engine.config.settings`:

```python
class AnalysisEngineSettings(AppSettings):
    # Add new parameters
    NEW_PARAMETER: str = Field(
        default="default_value",
        env="NEW_PARAMETER",
        description="Description"
    )

    # Add validation for the new parameter
    @field_validator('NEW_PARAMETER')
    @classmethod
    def validate_new_parameter(cls, v: str) -> str:
        """Validate the new parameter."""
        if not v.startswith("valid_"):
            raise ValueError(f"Invalid parameter value: {v}. Must start with 'valid_'")
        return v
```

### Adding Computed Fields

You can add computed fields that derive their values from other settings:

```python
class AnalysisEngineSettings(AppSettings):
    # Base parameters
    BASE_URL: str = Field(default="http://localhost", env="BASE_URL")
    API_PATH: str = Field(default="/api", env="API_PATH")
    API_VERSION: str = Field(default="v1", env="API_VERSION")

    # Computed field
    @computed_field(description="Full API URL")
    @property
    def API_URL(self) -> str:
        """Get the full API URL."""
        return f"{self.BASE_URL}{self.API_PATH}/{self.API_VERSION}"
```

### Adding Model Validators

You can add model validators to validate multiple fields together:

```python
class AnalysisEngineSettings(AppSettings):
    # Parameters
    MIN_VALUE: int = Field(default=10, env="MIN_VALUE")
    MAX_VALUE: int = Field(default=100, env="MAX_VALUE")

    # Model validator
    @model_validator(mode='after')
    def validate_min_max(self) -> 'AnalysisEngineSettings':
        """Validate that MIN_VALUE is less than MAX_VALUE."""
        if self.MIN_VALUE >= self.MAX_VALUE:
            raise ValueError(f"MIN_VALUE ({self.MIN_VALUE}) must be less than MAX_VALUE ({self.MAX_VALUE})")
        return self
```

## Testing

When writing tests that depend on configuration, use the `test_env_vars` fixture:

```python
def test_my_function(test_env_vars):
    # The test_env_vars fixture sets up test environment variables
    result = my_function()
    assert result == expected_value
```

### Testing with Custom Configuration

You can create a custom configuration for testing:

```python
from analysis_engine.config import AnalysisEngineSettings

def test_with_custom_config():
    # Create a custom configuration for testing
    test_config = AnalysisEngineSettings(
        DEBUG_MODE=True,
        LOG_LEVEL="DEBUG",
        DATABASE_URL_OVERRIDE="sqlite:///./test.db",
        RATE_LIMIT_REQUESTS=1000
    )

    # Use the custom configuration in your test
    result = my_function_that_uses_config(test_config)
    assert result == expected_value
```

### Mocking Configuration

You can mock the configuration for testing:

```python
from unittest.mock import patch
from analysis_engine.config import get_settings

@patch('analysis_engine.config.settings')
def test_with_mocked_config(mock_settings):
    # Configure the mock
    mock_settings.DEBUG_MODE = True
    mock_settings.LOG_LEVEL = "DEBUG"
    mock_settings.DATABASE_URL = "sqlite:///./test.db"

    # Use the mocked configuration in your test
    result = my_function_that_uses_config()
    assert result == expected_value
```

## Migration Guide

If you're currently using the deprecated configuration modules, follow these steps to migrate:

1. Replace imports from `analysis_engine.core.config` with `analysis_engine.config`:

   ```python
   # Before
   from analysis_engine.core.config import get_settings

   # After
   from analysis_engine.config import get_settings
   ```

2. Replace imports from `config.config` with `analysis_engine.config`:

   ```python
   # Before
   from config.config import API_VERSION, API_PREFIX

   # After
   from analysis_engine.config import get_settings
   settings = get_settings()
   API_VERSION = settings.API_VERSION
   API_PREFIX = settings.API_PREFIX
   ```

3. Update direct access to settings attributes:

   ```python
   # Before
   from analysis_engine.core.config import settings
   host = settings.host
   port = settings.port

   # After
   from analysis_engine.config import settings
   host = settings.HOST
   port = settings.PORT
   ```

4. Update helper function usage:

   ```python
   # Before
   from analysis_engine.core.config import get_db_settings
   db_settings = get_db_settings()

   # After
   from analysis_engine.config import get_db_settings
   db_settings = get_db_settings()
   # Note: The function name is the same, but it's imported from a different module
   ```

5. Update configuration manager usage:

   ```python
   # Before
   from analysis_engine.core.config import ConfigurationManager
   config_manager = ConfigurationManager()

   # After
   from analysis_engine.config import ConfigurationManager
   config_manager = ConfigurationManager()
   # Note: The class name is the same, but it's imported from a different module
   ```

## Best Practices

1. **Use Environment Variables**: Always use environment variables for configuration in production.
2. **Validate Configuration**: Use Pydantic's validation features to ensure configuration values are valid.
3. **Document Configuration**: Document all configuration parameters with descriptions and validation rules.
4. **Use Helper Functions**: Use helper functions to access related configuration parameters.
5. **Use Feature Flags**: Use feature flags to enable or disable features without changing code.
6. **Use Computed Fields**: Use computed fields to derive values from other settings.
7. **Use Model Validators**: Use model validators to validate multiple fields together.
8. **Test Configuration**: Write tests for configuration validation and usage.
9. **Use Type Hints**: Use type hints to ensure type safety.
10. **Use Constants**: Use constants for configuration keys to avoid typos.
