# Account Reconciliation Service Migration Guide

This guide helps you migrate from the original `account_reconciliation_service.py` module to the new modular account reconciliation package structure.

## Overview

The account reconciliation service has been refactored from a single monolithic file into a modular package structure. This improves maintainability, readability, and extensibility while maintaining backward compatibility.

## Backward Compatibility

The original `account_reconciliation_service.py` module has been replaced with a facade that delegates to the new modular implementation, so existing code that imports from the original module will continue to work without changes.

```python
# This still works
from portfolio_management_service.services.account_reconciliation_service import AccountReconciliationService
```

## Benefits of the New Structure

The new package structure provides several benefits:

1. **Better Organization**: Each reconciliation component is in its own module
2. **Improved Maintainability**: Smaller, focused files are easier to maintain
3. **Enhanced Functionality**: New utility functions and improved implementations
4. **Better Documentation**: Comprehensive documentation and examples
5. **Easier Testing**: Modular structure makes testing easier
6. **Performance Improvements**: Optimized implementations for better performance

## Migration Steps

### Step 1: Update Imports (Optional)

You can continue to import from the original module, but for better code organization, you can update your imports to use the new package structure:

```python
# Old import
from portfolio_management_service.services.account_reconciliation_service import AccountReconciliationService

# New import
from portfolio_management_service.services.account_reconciliation import AccountReconciliationService
```

### Step 2: Use Specialized Components (Optional)

The new package provides specialized components that weren't directly accessible in the original module:

```python
from portfolio_management_service.services.account_reconciliation.basic_reconciliation import BasicReconciliation
from portfolio_management_service.services.account_reconciliation.position_reconciliation import PositionReconciliation
from portfolio_management_service.services.account_reconciliation.full_reconciliation import FullReconciliation
from portfolio_management_service.services.account_reconciliation.historical_analysis import HistoricalAnalysis
from portfolio_management_service.services.account_reconciliation.reporting import ReconciliationReporting
from portfolio_management_service.services.account_reconciliation.discrepancy_handling import DiscrepancyHandling

# Create a specialized component
basic_reconciliation = BasicReconciliation(
    account_repository=account_repository,
    portfolio_repository=portfolio_repository,
    trading_gateway_client=trading_gateway_client,
    event_publisher=event_publisher,
    reconciliation_repository=reconciliation_repository
)

# Use the specialized component directly
result = await basic_reconciliation.perform_reconciliation(
    internal_data, broker_data, tolerance
)
```

### Step 3: Take Advantage of Enhanced Functionality

The new implementations provide enhanced functionality and better parameter handling:

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

# Use the new pattern detection functionality
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

## API Changes

There are no breaking changes to the public API. All methods from the original module are still available with the same signatures.

## Package Structure

The new package structure is organized as follows:

```
account_reconciliation/
├── __init__.py                # Re-exports all classes and functions
├── base.py                    # Base classes and common functionality
├── basic_reconciliation.py    # Basic reconciliation implementation
├── position_reconciliation.py # Position reconciliation implementation
├── full_reconciliation.py     # Full reconciliation implementation
├── historical_analysis.py     # Historical analysis implementation
├── reporting.py               # Reporting implementation
├── discrepancy_handling.py    # Discrepancy handling implementation
├── facade.py                  # Facade for all reconciliation functionality
├── README.md                  # Documentation and examples
└── MIGRATION_GUIDE.md         # This migration guide
```

## Questions and Support

If you have any questions or need support with the migration, please contact the platform development team.