# Configuration Management

This module provides a standardized configuration management system for the Forex Trading Platform. It supports loading configuration from files, environment variables, and defaults, with validation and type conversion.

## Key Components

1. **BaseAppSettings**: Base settings class with common configuration for all services
2. **ConfigManager**: Configuration manager for loading and accessing configuration
3. **ConfigSource**: Enumeration of configuration sources (file, environment, default, override)
4. **ConfigValue**: Configuration value with metadata (value, source, path)

## Usage

### Basic Usage

```python
from common_lib.config import BaseAppSettings, get_settings

# Get settings with default configuration
settings = get_settings()

# Access configuration values
db_host = settings.DB_HOST
db_port = settings.DB_PORT
log_level = settings.LOG_LEVEL

# Access computed properties
database_url = settings.database_url
redis_url = settings.redis_url
```

### Custom Settings Class

```python
from pydantic import Field
from common_lib.config import BaseAppSettings, get_settings

class MyServiceSettings(BaseAppSettings):
    # Override service name
    SERVICE_NAME: str = Field("my-service", description="Name of the service")
    
    # Add service-specific configuration
    MY_FEATURE_ENABLED: bool = Field(False, description="Enable my feature")
    MY_FEATURE_TIMEOUT: int = Field(30, description="My feature timeout in seconds")
    
    # Add validation
    @field_validator("MY_FEATURE_TIMEOUT")
    def validate_timeout(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Timeout must be non-negative")
        return v

# Get settings with custom settings class
settings = get_settings(settings_class=MyServiceSettings)
```

### Configuration Manager

```python
from common_lib.config import StandardizedConfigManager, BaseAppSettings

# Create a configuration manager
config_manager = StandardizedConfigManager(
    settings_class=BaseAppSettings,
    env_file=".env",
    config_file="config.yaml",
    env_prefix="MY_SERVICE_"
)

# Get a configuration value
db_host = config_manager.get("DB_HOST")

# Get all configuration values
all_config = config_manager.get_all()

# Get a configuration value with metadata
db_host_with_metadata = config_manager.get_with_metadata("DB_HOST")
print(f"Value: {db_host_with_metadata.value}")
print(f"Source: {db_host_with_metadata.source}")
print(f"Path: {db_host_with_metadata.path}")

# Override a configuration value
config_manager.override("LOG_LEVEL", "DEBUG")

# Reload configuration
config_manager.reload()
```

### Cached Configuration Manager

```python
from common_lib.config import get_config_manager, BaseAppSettings

# Get a cached configuration manager
config_manager = get_config_manager(
    settings_class=BaseAppSettings,
    env_file=".env",
    config_file="config.yaml",
    env_prefix="MY_SERVICE_"
)

# Access the settings instance
settings = config_manager.settings
```

## Configuration Sources

The configuration system loads configuration from multiple sources in the following order of precedence:

1. **Override**: Configuration values set programmatically using `override()`
2. **Environment Variables**: Configuration values set in environment variables
3. **Configuration File**: Configuration values set in the configuration file
4. **Default Values**: Default values defined in the settings class

## Environment Variables

Environment variables are automatically mapped to configuration fields. For example:

- `DB_HOST` environment variable maps to `DB_HOST` field
- `DB_PORT` environment variable maps to `DB_PORT` field

If an environment prefix is specified, it is prepended to the field name:

- With prefix `MY_SERVICE_`, `MY_SERVICE_DB_HOST` maps to `DB_HOST` field
- With prefix `MY_SERVICE_`, `MY_SERVICE_DB_PORT` maps to `DB_PORT` field

## Configuration Files

The configuration system supports loading configuration from YAML and JSON files. For example:

```yaml
# config.yaml
SERVICE_NAME: my-service
ENVIRONMENT: production
LOG_LEVEL: INFO
DB_HOST: db.example.com
DB_PORT: 5432
DB_NAME: mydb
DB_USER: myuser
DB_PASSWORD: mypassword
```

```json
// config.json
{
  "SERVICE_NAME": "my-service",
  "ENVIRONMENT": "production",
  "LOG_LEVEL": "INFO",
  "DB_HOST": "db.example.com",
  "DB_PORT": 5432,
  "DB_NAME": "mydb",
  "DB_USER": "myuser",
  "DB_PASSWORD": "mypassword"
}
```

## Validation

The configuration system validates configuration values using Pydantic's validation system. For example:

- `ENVIRONMENT` must be one of "development", "testing", "staging", or "production"
- `LOG_LEVEL` must be one of "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL"
- `DB_PORT` must be between 1 and 65535
- `DB_POOL_SIZE` must be at least 1

## Computed Properties

The configuration system supports computed properties that are derived from other configuration values. For example:

- `database_url` is computed from `DB_DRIVER`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, and `DB_NAME`
- `redis_url` is computed from `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, and `REDIS_PASSWORD`

## Migration Guide

If you are migrating from the legacy configuration system to the standardized configuration system, follow these steps:

1. **Update Imports**:
   ```python
   # Legacy
   from common_lib.config import ConfigManager
   
   # Standardized
   from common_lib.config import StandardizedConfigManager, BaseAppSettings
   ```

2. **Create Custom Settings Class**:
   ```python
   # Legacy
   class MyServiceConfig(ServiceSpecificConfig):
       my_feature_enabled: bool = False
       my_feature_timeout: int = 30
   
   # Standardized
   class MyServiceSettings(BaseAppSettings):
       MY_FEATURE_ENABLED: bool = Field(False, description="Enable my feature")
       MY_FEATURE_TIMEOUT: int = Field(30, description="My feature timeout in seconds")
   ```

3. **Initialize Configuration Manager**:
   ```python
   # Legacy
   config_manager = ConfigManager(
       config_path="config/config.yaml",
       service_specific_model=MyServiceConfig,
       env_prefix="MY_SERVICE_",
       default_config_path="config/default_config.yaml"
   )
   
   # Standardized
   config_manager = StandardizedConfigManager(
       settings_class=MyServiceSettings,
       env_file=".env",
       config_file="config/config.yaml",
       env_prefix="MY_SERVICE_"
   )
   ```

4. **Access Configuration Values**:
   ```python
   # Legacy
   service_config = config_manager.get_service_specific_config()
   my_feature_enabled = service_config.my_feature_enabled
   
   # Standardized
   settings = config_manager.settings
   my_feature_enabled = settings.MY_FEATURE_ENABLED
   ```
