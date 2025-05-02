# Validation Components Documentation

This document explains the purpose and relationship between different validation components in the data-pipeline-service.

## Validation Engine Components

The service contains two primary validation engines:

### 1. `validation_engine.py` (Base Implementation)
- Basic validator interfaces and concrete validators for common cases
- Lightweight implementation suitable for simple validation scenarios
- Used for basic data integrity checks and common validation rules

### 2. `advanced_validation_engine.py` (Extended Implementation)
- Built on top of the base validators
- Implements more sophisticated validation strategies
- Uses machine learning techniques for anomaly detection
- Handles complex rule combinations and validation pipelines
- Should be used for advanced use cases where basic validation is insufficient

## Usage Guidelines

- For simple validation scenarios, use the base `validation_engine.py` components
- For complex or ML-based validation, use the `advanced_validation_engine.py` components
- When extending the validation system, add new basic validators to `validation_engine.py` and advanced validators to `advanced_validation_engine.py`

## Future Development

The long-term plan is to maintain both implementations:
- The basic implementation for performance-sensitive and simple validation cases
- The advanced implementation for complex data quality assurance

There is no plan to deprecate either implementation as they serve different purposes in the validation pipeline.
