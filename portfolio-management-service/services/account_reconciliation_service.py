"""
Account Reconciliation Service

This module provides functionality for automatic reconciliation between internal account data
and data fetched from trading brokers via the Trading Gateway Service, ensuring data integrity
and accuracy.

Note: This module has been refactored into a modular package structure.
      It now imports from the account_reconciliation/ package to maintain backward compatibility.
      See account_reconciliation/README.md for more information.
"""

# Re-export the AccountReconciliationService from the new package
from services.facade import AccountReconciliationService

# Define all exports
__all__ = ['AccountReconciliationService']