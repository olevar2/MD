# Data Reconciliation Framework

## Overview

The Data Reconciliation Framework provides a comprehensive solution for reconciling data between different sources in the Forex Trading Platform. It supports both real-time and batch reconciliation processes, with configurable resolution strategies for handling discrepancies.

## Key Features

- **Flexible Source Configuration**: Support for multiple data sources with configurable priorities
- **Customizable Resolution Strategies**: Multiple strategies for resolving discrepancies, including source priority, most recent, average, median, and custom strategies
- **Batch and Real-time Processing**: Support for both batch reconciliation of historical data and real-time reconciliation of streaming data
- **Comprehensive Reporting**: Detailed reports on reconciliation results, including discrepancies, resolutions, and metrics
- **Service-specific Implementations**: Tailored implementations for different services in the platform

## Architecture

The framework is organized into the following components:

### Core Components

- **Base Classes and Interfaces**: Foundational classes for data reconciliation
- **Resolution Strategies**: Strategies for resolving discrepancies between data sources
- **Batch Reconciliation**: Implementations for batch reconciliation processes
- **Real-time Reconciliation**: Implementations for real-time reconciliation processes
- **Reporting Utilities**: Utilities for generating reports and metrics from reconciliation results
- **Custom Exceptions**: Specialized exceptions for error handling

### Service-specific Implementations

- **Data Pipeline Service**: Reconciliation for market data, including OHLCV and tick data
- **Feature Store Service**: Reconciliation for feature data, including feature versions and feature values
- **ML Integration Service**: Reconciliation for model data, including training and inference data

## Usage Examples

### Basic Reconciliation

```python
from common_lib.data_reconciliation import (
    DataSource,
    DataSourceType,
    ReconciliationConfig,
    ReconciliationStrategy,
    BatchReconciliationProcessor,
)

# Define data sources
source1 = DataSource(
    source_id="database",
    name="Database",
    source_type=DataSourceType.DATABASE,
    priority=1
)

source2 = DataSource(
    source_id="api",
    name="External API",
    source_type=DataSourceType.API,
    priority=2
)

# Create configuration
config = ReconciliationConfig(
    sources=[source1, source2],
    strategy=ReconciliationStrategy.SOURCE_PRIORITY,
    tolerance=0.0001,
    auto_resolve=True
)

# Create reconciliation processor
reconciliation = BatchReconciliationProcessor(config)

# Define data fetchers
async def fetch_from_database(**kwargs):
    # Fetch data from database
    return {"key1": 100, "key2": 200}

async def fetch_from_api(**kwargs):
    # Fetch data from API
    return {"key1": 101, "key2": 200}

reconciliation.data_fetchers = {
    "database": fetch_from_database,
    "api": fetch_from_api,
}

# Perform reconciliation
result = await reconciliation.reconcile()

# Process results
print(f"Discrepancies found: {result.discrepancy_count}")
print(f"Resolutions applied: {result.resolution_count}")
```

### Market Data Reconciliation

```python
from data_pipeline_service.reconciliation import OHLCVReconciliation
from common_lib.data_reconciliation import (
    DataSource,
    DataSourceType,
    ReconciliationConfig,
    ReconciliationStrategy,
)

# Define data sources
source1 = DataSource(
    source_id="primary_provider",
    name="Primary Data Provider",
    source_type=DataSourceType.API,
    priority=2
)

source2 = DataSource(
    source_id="secondary_provider",
    name="Secondary Data Provider",
    source_type=DataSourceType.API,
    priority=1
)

# Create configuration
config = ReconciliationConfig(
    sources=[source1, source2],
    strategy=ReconciliationStrategy.SOURCE_PRIORITY,
    tolerance=0.0001,
    auto_resolve=True
)

# Create OHLCV reconciliation
reconciliation = OHLCVReconciliation(
    config=config,
    data_fetcher_manager=data_fetcher_manager,
    validation_engine=validation_engine,
    ohlcv_repository=ohlcv_repository
)

# Perform reconciliation
result = await reconciliation.reconcile(
    symbol="EUR_USD",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 1, 31),
    timeframe="1h"
)

# Generate report
report = ReconciliationReport(result)
print(report.to_dict())
```

### Real-time Reconciliation

```python
from common_lib.data_reconciliation import (
    DataSource,
    DataSourceType,
    ReconciliationConfig,
    ReconciliationStrategy,
    StreamingDataReconciliation,
)

# Define data sources
source1 = DataSource(
    source_id="primary_feed",
    name="Primary Data Feed",
    source_type=DataSourceType.STREAM,
    priority=2
)

source2 = DataSource(
    source_id="secondary_feed",
    name="Secondary Data Feed",
    source_type=DataSourceType.STREAM,
    priority=1
)

# Create configuration
config = ReconciliationConfig(
    sources=[source1, source2],
    strategy=ReconciliationStrategy.MOST_RECENT,
    tolerance=0.0001,
    auto_resolve=True
)

# Create streaming reconciliation
reconciliation = StreamingDataReconciliation(
    config=config,
    reconciliation_interval=1.0  # Reconcile every second
)

# Define data fetchers and updaters
async def fetch_from_primary(**kwargs):
    # Fetch latest data from primary feed
    return {"price": 1.1234, "volume": 1000}

async def fetch_from_secondary(**kwargs):
    # Fetch latest data from secondary feed
    return {"price": 1.1235, "volume": 1005}

async def update_primary(field, value):
    # Update primary feed with resolved value
    print(f"Updating primary feed: {field} = {value}")
    return True

async def update_secondary(field, value):
    # Update secondary feed with resolved value
    print(f"Updating secondary feed: {field} = {value}")
    return True

reconciliation.data_fetchers = {
    "primary_feed": fetch_from_primary,
    "secondary_feed": fetch_from_secondary,
}

reconciliation.data_updaters = {
    "primary_feed": update_primary,
    "secondary_feed": update_secondary,
}

# Start reconciliation
await reconciliation.start()

# Stop reconciliation after some time
await asyncio.sleep(60)
await reconciliation.stop()
```

## Resolution Strategies

The framework provides several strategies for resolving discrepancies:

### Source Priority Strategy

Uses data from the highest priority source. This is useful when one source is considered more authoritative than others.

### Most Recent Strategy

Uses the most recently updated data. This is useful for real-time data where the most recent value is typically the most accurate.

### Average Values Strategy

Uses the average of all values. This is useful for numeric data where all sources are equally reliable.

### Median Values Strategy

Uses the median of all values. This is useful for numeric data where there may be outliers.

### Custom Resolution Strategy

Uses a custom function to resolve discrepancies. This allows for domain-specific resolution logic.

### Threshold-based Strategy

Uses different strategies based on the magnitude of the discrepancy. For example, small discrepancies might be resolved using the average, while large discrepancies might require using the most authoritative source.

## Reporting

The framework provides comprehensive reporting capabilities:

### Reconciliation Report

Detailed report on a single reconciliation process, including discrepancies, resolutions, and metrics.

### Discrepancy Report

Focused report on discrepancies, including severity, sources, and resolution status.

### Reconciliation Summary

Summary of multiple reconciliation processes over time, including trends and metrics.

### Reconciliation Metrics

Statistical metrics calculated from reconciliation results, such as resolution rate, severity distribution, and performance metrics.

## Implementation Details

### Data Sources

Data sources are defined using the `DataSource` class, which includes:

- **Source ID**: Unique identifier for the source
- **Name**: Human-readable name of the source
- **Source Type**: Type of the data source (database, API, file, stream, etc.)
- **Priority**: Priority of the source (higher number = higher priority)
- **Metadata**: Additional metadata about the source

### Discrepancies

Discrepancies are represented by the `Discrepancy` class, which includes:

- **Field**: Field or attribute where the discrepancy was found
- **Sources**: Dictionary mapping source IDs to their values
- **Severity**: Severity level of the discrepancy
- **Statistics**: Calculated statistics about the discrepancy (min, max, mean, median, etc.)

### Resolutions

Resolutions are represented by the `DiscrepancyResolution` class, which includes:

- **Discrepancy**: The discrepancy being resolved
- **Resolved Value**: The value used to resolve the discrepancy
- **Strategy**: Strategy used to resolve the discrepancy
- **Resolution Source**: Source of the resolved value (if applicable)

## Service Integration

The Data Reconciliation Framework has been integrated with the following services:

- **Data Pipeline Service**: Reconciliation for market data
- **Feature Store Service**: Reconciliation for feature data
- **ML Integration Service**: Reconciliation for model data

Each service has specialized implementations tailored to its specific data types and requirements.

## Future Improvements

1. **Additional Resolution Strategies**: Implement more sophisticated resolution strategies, such as weighted averages and machine learning-based approaches
2. **Enhanced Visualization**: Create visualization tools for reconciliation results and trends
3. **Automated Anomaly Detection**: Implement anomaly detection for identifying unusual discrepancies
4. **Integration with Monitoring System**: Connect with the platform's monitoring and alerting system
5. **Performance Optimization**: Optimize for high-throughput reconciliation of large datasets
