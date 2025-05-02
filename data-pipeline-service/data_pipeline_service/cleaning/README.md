# Data Cleaning Components Documentation

This document explains the purpose and relationship between different data cleaning components in the data-pipeline-service.

## Cleaning Engine Components

The service contains two primary cleaning engines:

### 1. `cleaning_engine.py` (Base Implementation)
- Core cleaning interfaces and basic implementations
- Handles common cleaning tasks like missing value imputation and outlier removal
- Optimized for performance and minimal dependencies
- Used for standard data preprocessing workflows

### 2. `advanced_cleaning_engine.py` (Enhanced Implementation)
- Extends the base cleaning capabilities with sophisticated algorithms
- Implements ML-based cleaning strategies using scikit-learn
- Provides specialized cleaning for financial time series data
- Includes advanced anomaly correction and distribution-aware imputation
- Has additional dependencies on machine learning libraries

## Usage Guidelines

- For standard cleaning tasks with minimal dependencies, use the base `cleaning_engine.py`
- When more sophisticated cleaning is required, use `advanced_cleaning_engine.py`
- The cleaning pipeline can use both engines in different stages based on requirements

## Future Development

Both engines will be maintained as they serve different use cases:
- `cleaning_engine.py` for performance-critical and basic cleaning needs
- `advanced_cleaning_engine.py` for specialized cleaning scenarios requiring advanced techniques

There is no plan to merge these components as they have different dependency profiles and performance characteristics.
