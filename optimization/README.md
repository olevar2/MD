# Optimization Module

This module contains optimization utilities for the Forex Trading Platform, including:

- Resource allocation algorithms
- ML-based optimization techniques
- Time series optimization functions
- Caching strategies

## Installation

```bash
# From the optimization directory
poetry install
```

## Usage

The optimization module provides various utilities for optimizing different aspects of the trading platform:

```python
from optimization import resource_allocator
from optimization.caching import strategy_cache
from optimization.ml import hyperparameter_tuning
```

## Testing

```bash
# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=optimization
```
