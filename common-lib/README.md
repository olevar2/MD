# Common Library

A shared library for common utilities and functionality used across the forex trading platform services.

## Purpose

This library centralizes commonly used code to reduce duplication and ensure consistency across all microservices within the platform.

## Components

- **exceptions.py**: Common exception classes used throughout the platform
- **security.py**: Shared security mechanisms including authentication utilities
- **resilience/**: Resilience patterns (circuit breaker, retry, timeout, bulkhead) for robust service communication

## Usage

Add this library as a path dependency in your service's pyproject.toml:

```toml
[tool.poetry.dependencies]
common-lib = {path = "../common-lib", develop = true}
```

Then import and use the components as needed:

```python
from common_lib.exceptions import ValidationError
from common_lib.security import validate_api_key

# Resilience patterns
from common_lib.resilience import (
    retry_with_policy,
    timeout_handler, 
    bulkhead,
    create_circuit_breaker
)
```

See the `usage_demos` directory for complete working examples.
