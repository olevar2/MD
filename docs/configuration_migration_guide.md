# Configuration Migration Guide

## Overview

The Analysis Engine Service has consolidated its configuration system into a single module. This guide will help you migrate from the deprecated configuration modules to the new consolidated module.

## Timeline

- **Now - December 31, 2023**: Transition period
  - Both old and new configuration modules are available
  - Deprecation warnings are shown when using old modules
  - Usage of deprecated modules is monitored
- **After December 31, 2023**: Removal
  - Deprecated modules will be removed
  - Code using deprecated modules will break

## Migration Steps

### 1. Migrating from `analysis_engine.core.config`

#### Before:

```python
from analysis_engine.core.config import Settings, get_settings, ConfigurationManager

# Get settings instance
settings = get_settings()

# Access settings
host = settings.host
port = settings.port

# Use configuration manager
config_manager = ConfigurationManager()
value = config_manager.get("host")
```

#### After:

```python
from analysis_engine.config import AnalysisEngineSettings, get_settings, ConfigurationManager

# Get settings instance
settings = get_settings()

# Access settings (note the uppercase attribute names)
host = settings.HOST
port = settings.PORT

# Use configuration manager
config_manager = ConfigurationManager()
value = config_manager.get("HOST")  # Note the uppercase key
```

#### Key Changes:

1. Import from `analysis_engine.config` instead of `analysis_engine.core.config`
2. Use `AnalysisEngineSettings` instead of `Settings`
3. Settings attributes are now UPPERCASE (e.g., `HOST` instead of `host`)
4. Configuration manager keys are now UPPERCASE (e.g., `"HOST"` instead of `"host"`)

### 2. Migrating from `config.config`

#### Before:

```python
from config.config import API_VERSION, API_PREFIX, get_settings

# Access settings directly
api_version = API_VERSION
api_prefix = API_PREFIX

# Get all settings
all_settings = get_settings()
```

#### After:

```python
from analysis_engine.config import settings, get_settings

# Access settings through the settings object
api_version = settings.API_VERSION
api_prefix = settings.API_PREFIX

# Get all settings
all_settings = get_settings().__dict__
```

#### Key Changes:

1. Import from `analysis_engine.config` instead of `config.config`
2. Access settings through the `settings` object
3. Use `get_settings().__dict__` to get all settings as a dictionary

### 3. Migrating Helper Functions

#### Before:

```python
from analysis_engine.core.config import get_db_settings, get_api_settings

# Get database settings
db_settings = get_db_settings()
database_url = db_settings["database_url"]

# Get API settings
api_settings = get_api_settings()
host = api_settings["host"]
```

#### After:

```python
from analysis_engine.config import get_db_settings, get_api_settings

# Get database settings
db_settings = get_db_settings()
database_url = db_settings["database_url"]

# Get API settings
api_settings = get_api_settings()
host = api_settings["host"]
```

#### Key Changes:

1. Import from `analysis_engine.config` instead of `analysis_engine.core.config`
2. Function names and return values remain the same

## Testing Your Migration

After migrating, you should test your code to ensure it works correctly with the new configuration system. Here are some tips:

1. Run your tests with deprecation warnings enabled:
   ```bash
   python -Wd your_test_script.py
   ```

2. Check for any deprecation warnings related to configuration modules

3. Verify that your code works correctly with the new configuration system

## Monitoring Deprecation Usage

We have implemented a monitoring system to track usage of deprecated modules. You can generate a report using the following command:

```bash
python tools/deprecation_report.py --format html --output deprecation_report.html
```

This will generate a report showing where deprecated modules are still being used in the codebase.

## Getting Help

If you encounter any issues during migration, please contact the platform team for assistance.

## FAQ

### Q: Why are we migrating to a new configuration system?

A: The new configuration system provides several benefits:
- Single source of truth for configuration
- Type safety and validation
- Better documentation
- Consistent patterns across services

### Q: Will my code break after migration?

A: If you follow the migration steps, your code should continue to work. The main changes are in import paths and attribute names.

### Q: What if I miss the migration deadline?

A: After December 31, 2023, the deprecated modules will be removed, and code that still uses them will break. It's important to complete the migration before this date.

### Q: How can I verify that I've migrated everything?

A: Run your code with deprecation warnings enabled (`python -Wd your_script.py`) and check for any warnings related to configuration modules. You can also use the deprecation report tool to identify usage of deprecated modules.

### Q: Can I use both old and new configuration modules during the transition?

A: Yes, during the transition period, both old and new modules are available. However, you should migrate to the new module as soon as possible to avoid issues when the old modules are removed.
