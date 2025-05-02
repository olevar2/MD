# Configuration System Enhancements

## Overview

This document details the enhancements made to the Analysis Engine Service's configuration system. These improvements focus on consolidating configuration management, improving validation, and providing better documentation.

## Key Enhancements

### 1. Service Code Updates

We've updated service code to use the consolidated configuration module directly:

- **Core Module Updates**:
  - Updated `analysis_engine/core/logging.py` to import from `analysis_engine.config`
  - Updated `analysis_engine/core/database.py` to import from `analysis_engine.config`
  - Updated `analysis_engine/core/connection_pool.py` to import from `analysis_engine.config`
  - Updated `main.py` to import from `analysis_engine.config`
  - Updated `tests/core/test_config.py` to import from `analysis_engine.config`

- **Import Pattern**:
  ```python
  # Before
  from analysis_engine.core.config import get_settings
  
  # After
  from analysis_engine.config import get_settings
  ```

- **Class Name Updates**:
  ```python
  # Before
  from analysis_engine.core.config import Settings
  
  # After
  from analysis_engine.config import AnalysisEngineSettings as Settings
  ```

### 2. Enhanced Validation Rules

We've significantly improved the validation rules in the configuration module:

#### API Settings Validation

```python
@field_validator('API_VERSION')
@classmethod
def validate_api_version(cls, v: str) -> str:
    """Validate API version format."""
    if not re.match(r'^v\d+(\.\d+)?$', v):
        raise ValueError(f"Invalid API version format: {v}. Must be in format 'v1' or 'v1.0'")
    return v

@field_validator('API_PREFIX')
@classmethod
def validate_api_prefix(cls, v: str) -> str:
    """Validate API prefix format."""
    if not v.startswith('/'):
        v = f"/{v}"
    return v

@field_validator('HOST')
@classmethod
def validate_host(cls, v: str) -> str:
    """Validate host address."""
    if v != "localhost" and v != "0.0.0.0":
        try:
            IPv4Address(v)
        except ValueError:
            try:
                IPv6Address(v)
            except ValueError:
                raise ValueError(f"Invalid host address: {v}")
    return v
```

#### Security Settings Validation

```python
@field_validator('JWT_SECRET')
@classmethod
def validate_jwt_secret(cls, v: str) -> str:
    """Validate JWT secret."""
    if v == "your-secret-key":
        import warnings
        warnings.warn(
            "Using default JWT_SECRET is not secure. Please set a proper secret key.",
            UserWarning,
            stacklevel=2
        )
    if len(v) < 16:
        raise ValueError("JWT_SECRET must be at least 16 characters long")
    return v

@field_validator('JWT_ALGORITHM')
@classmethod
def validate_jwt_algorithm(cls, v: str) -> str:
    """Validate JWT algorithm."""
    allowed_algorithms = ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512", "ES256", "ES384", "ES512"]
    if v not in allowed_algorithms:
        raise ValueError(f"Invalid JWT algorithm: {v}. Must be one of {allowed_algorithms}")
    return v
```

#### Database Settings Validation

```python
@field_validator('DATABASE_URL_OVERRIDE')
@classmethod
def validate_database_url(cls, v: Optional[str]) -> Optional[str]:
    """Validate database URL format."""
    if v is None:
        return v
        
    valid_prefixes = [
        "postgresql://", "postgresql+psycopg2://", "postgresql+asyncpg://",
        "mysql://", "mysql+pymysql://", "mysql+aiomysql://",
        "sqlite://", "oracle://", "mssql://", "cockroachdb://"
    ]
    
    if not any(v.startswith(prefix) for prefix in valid_prefixes):
        raise ValueError(f"Invalid database URL format: {v}. Must start with one of {valid_prefixes}")
    
    return v

@model_validator(mode='after')
def validate_db_settings(self) -> 'AnalysisEngineSettings':
    """Validate database settings consistency."""
    # If DATABASE_URL_OVERRIDE is provided, we don't need to validate individual components
    if self.DATABASE_URL_OVERRIDE:
        return self
        
    # If any of the required DB components are provided, all must be provided
    db_components = [self.DB_USER, self.DB_HOST, self.DB_NAME]
    if any(db_components) and not all(db_components):
        missing = []
        if not self.DB_USER:
            missing.append("DB_USER")
        if not self.DB_HOST:
            missing.append("DB_HOST")
        if not self.DB_NAME:
            missing.append("DB_NAME")
            
        raise ValueError(f"Missing required database settings: {', '.join(missing)}")
        
    return self
```

#### Redis Settings Validation

```python
@field_validator('REDIS_HOST')
@classmethod
def validate_redis_host(cls, v: str) -> str:
    """Validate Redis host."""
    if v != "localhost":
        try:
            IPv4Address(v)
        except ValueError:
            try:
                IPv6Address(v)
            except ValueError:
                # Allow domain names
                if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$', v):
                    raise ValueError(f"Invalid Redis host: {v}")
    return v
```

#### External Service Validation

```python
@field_validator('MARKET_DATA_SERVICE_URL', 'NOTIFICATION_SERVICE_URL', 'ML_INTEGRATION_SERVICE_URL', 'FEATURE_STORE_SERVICE_URL')
@classmethod
def validate_service_url(cls, v: str) -> str:
    """Validate service URL format."""
    if not v.startswith(("http://", "https://")):
        raise ValueError(f"Invalid service URL: {v}. Must start with http:// or https://")
    return v
```

#### Analysis Settings Validation

```python
@field_validator('DEFAULT_TIMEFRAMES')
@classmethod
def validate_timeframes(cls, v: List[str]) -> List[str]:
    """Validate timeframe format."""
    valid_timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN"]
    for timeframe in v:
        if timeframe not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}")
    return v

@field_validator('DEFAULT_SYMBOLS')
@classmethod
def validate_symbols(cls, v: List[str]) -> List[str]:
    """Validate symbol format."""
    # Basic validation for forex symbols
    for symbol in v:
        if not re.match(r'^[A-Z]{6}$', symbol):
            raise ValueError(f"Invalid forex symbol: {symbol}. Must be 6 uppercase letters (e.g., EURUSD)")
    return v
```

### 3. Extended Documentation

We've significantly enhanced the documentation with more examples and use cases:

#### Advanced Usage Examples

```python
# Using Feature Flags
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

```python
# Configuring External Service Connections
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

#### Detailed Environment Variables Documentation

We've organized environment variables by category with validation rules:

| Environment Variable | Description | Default Value | Validation |
|---------------------|-------------|---------------|------------|
| `API_VERSION` | API version | `"v1"` | Format: v1, v1.0, etc. |
| `API_PREFIX` | API endpoint prefix | `"/api/v1"` | Must start with / |
| `HOST` | Host to bind the API server | `"0.0.0.0"` | Valid IP address or hostname |
| `PORT` | Port to bind the API server | `8000` | Range: 1024-65535 |

#### Testing Examples

```python
# Testing with Custom Configuration
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

```python
# Mocking Configuration
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

### 4. Enhanced Deprecation Warnings

We've improved the deprecation warnings in the old configuration modules:

```python
def _show_deprecation_warning():
    # Get the caller's frame
    frame = inspect.currentframe().f_back.f_back
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    function_name = frame.f_code.co_name
    
    # Get relative path for better readability
    try:
        rel_path = os.path.relpath(filename)
    except ValueError:
        rel_path = filename
    
    # Record usage for monitoring
    record_usage("analysis_engine.core.config")
    
    warnings.warn(
        f"DEPRECATION WARNING: analysis_engine.core.config will be removed after {REMOVAL_DATE} ({days_message}).\n"
        f"Please import from analysis_engine.config instead.\n"
        f"Called from {rel_path}:{lineno} in function '{function_name}'\n"
        f"Migration guide: https://confluence.example.com/display/DEV/Configuration+Migration+Guide",
        DeprecationWarning,
        stacklevel=2
    )
```

## Benefits of These Enhancements

1. **Improved Type Safety**: The enhanced validation rules catch configuration errors early
2. **Better Documentation**: Comprehensive documentation makes the system easier to understand and use
3. **Centralized Configuration**: A single source of truth for configuration simplifies maintenance
4. **Smooth Migration Path**: Clear deprecation warnings and migration tools help with the transition

## Migration Path

The enhancements include a clear migration path:

1. **Update Imports**: Change imports to use the new module
   ```python
   # Before
   from analysis_engine.core.config import get_settings
   
   # After
   from analysis_engine.config import get_settings
   ```

2. **Update Class Names**: Use the new class names
   ```python
   # Before
   from analysis_engine.core.config import Settings
   
   # After
   from analysis_engine.config import AnalysisEngineSettings as Settings
   ```

3. **Update Attribute Names**: Use uppercase attribute names
   ```python
   # Before
   host = settings.host
   
   # After
   host = settings.HOST
   ```

## Conclusion

These enhancements significantly improve the configuration system of the Analysis Engine Service. The improved validation, better documentation, and clear migration path make the system more robust and easier to use.

The changes maintain backward compatibility while encouraging migration to the new consolidated module, ensuring a smooth transition for all services that depend on the configuration system.
