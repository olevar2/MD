# Account Reconciliation Service

This package provides comprehensive account reconciliation functionality for the Forex Trading Platform.

## Overview

Account reconciliation is a critical process that ensures consistency between internal account data and external broker data. This package implements various reconciliation tools including:

- **Basic Reconciliation**: Reconcile account balance and equity
- **Position Reconciliation**: Reconcile positions between internal and broker data
- **Full Reconciliation**: Comprehensive reconciliation including orders
- **Historical Analysis**: Analyze reconciliation data over time to identify patterns and trends
- **Discrepancy Handling**: Handle reconciliation discrepancies with automatic or manual resolution
- **Reporting**: Generate reports from reconciliation data

## Usage Examples

### Basic Account Reconciliation

```python
from portfolio_management_service.services.account_reconciliation import AccountReconciliationService

# Create the reconciliation service
reconciliation_service = AccountReconciliationService(
    account_repository=account_repository,
    portfolio_repository=portfolio_repository,
    trading_gateway_client=trading_gateway_client,
    event_publisher=event_publisher,
    reconciliation_repository=reconciliation_repository
)

# Perform basic reconciliation
result = await reconciliation_service.reconcile_account(
    account_id="12345",
    reconciliation_level="basic",
    tolerance=0.01,
    notification_threshold=1.0,
    auto_fix=False
)

# Check for discrepancies
if result["discrepancies"]["total_count"] > 0:
    print(f"Found {result['discrepancies']['total_count']} discrepancies")
    for disc in result["discrepancies"]["details"]:
        print(f"Field: {disc['field']}, Internal: {disc['internal_value']}, Broker: {disc['broker_value']}")
```

### Position Reconciliation

```python
# Perform position reconciliation
result = await reconciliation_service.reconcile_account(
    account_id="12345",
    reconciliation_level="positions",
    tolerance=0.01,
    notification_threshold=1.0,
    auto_fix=False
)

# Check for position discrepancies
position_discrepancies = [d for d in result["discrepancies"]["details"] if "position" in d["field"]]
if position_discrepancies:
    print(f"Found {len(position_discrepancies)} position discrepancies")
```

### Full Reconciliation

```python
# Perform full reconciliation
result = await reconciliation_service.reconcile_account(
    account_id="12345",
    reconciliation_level="full",
    tolerance=0.01,
    notification_threshold=1.0,
    auto_fix=True  # Automatically fix low severity discrepancies
)
```

### Historical Analysis

```python
# Perform historical reconciliation analysis
from datetime import datetime, timedelta

end_date = datetime.utcnow()
start_date = end_date - timedelta(days=30)

analysis = await reconciliation_service.perform_historical_reconciliation_analysis(
    account_id="12345",
    start_date=start_date,
    end_date=end_date,
    interval="daily",
    reconciliation_level="basic",
    tolerance=0.01
)

# Check for trends
if analysis["analysis"]["trend"]["status"] == "success":
    print(f"Trend direction: {analysis['analysis']['trend']['direction']}")
    print(f"Trend strength: {analysis['analysis']['trend']['strength']}")
```

### Detect Patterns

```python
# Detect reconciliation patterns
patterns = await reconciliation_service.detect_reconciliation_patterns(
    account_id="12345",
    lookback_days=90
)

# Check for recurring patterns
if patterns["patterns_detected"]:
    print("Detected patterns:")
    for pattern in patterns["recurring_patterns"]:
        print(f"Field: {pattern['field']}, Frequency: {pattern['frequency_percentage']}%, Criticality: {pattern['criticality']}")
```

### Generate Reports

```python
# Generate a summary report
report = await reconciliation_service.get_historical_reconciliation_report(
    account_id="12345",
    start_date=start_date,
    end_date=end_date,
    report_format="summary"
)

# Generate a detailed report
detailed_report = await reconciliation_service.get_historical_reconciliation_report(
    account_id="12345",
    start_date=start_date,
    end_date=end_date,
    report_format="detailed"
)

# Generate a chart report
chart_report = await reconciliation_service.get_historical_reconciliation_report(
    account_id="12345",
    start_date=start_date,
    end_date=end_date,
    report_format="chart"
)
```

## Architecture

The package is organized into the following modules:

- `__init__.py`: Re-exports all reconciliation classes from the facade
- `base.py`: Base classes and common functionality
- `basic_reconciliation.py`: Basic reconciliation implementation
- `position_reconciliation.py`: Position reconciliation implementation
- `full_reconciliation.py`: Full reconciliation implementation
- `historical_analysis.py`: Historical analysis implementation
- `reporting.py`: Reporting implementation
- `discrepancy_handling.py`: Discrepancy handling implementation
- `facade.py`: Facade for all reconciliation functionality

All classes and functions are re-exported from the package root for easy access.

## Migration Guide

If you were using the original `account_reconciliation_service.py` module, you can continue to use it as before. The original module has been replaced with a facade that delegates to the new modular implementation, maintaining backward compatibility.

To migrate to the new package structure, simply update your imports:

```python
# Old import
from portfolio_management_service.services.account_reconciliation_service import AccountReconciliationService

# New import
from portfolio_management_service.services.account_reconciliation import AccountReconciliationService
```

The new package structure provides better organization, more functionality, and improved performance.