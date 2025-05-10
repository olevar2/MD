"""
Data Reconciliation Framework.

This module provides a comprehensive framework for reconciling data between different sources,
supporting both real-time and batch reconciliation processes.

The framework includes:
- Base classes and interfaces for data reconciliation
- Resolution strategies for handling conflicting data
- Batch and real-time reconciliation implementations
- Reporting utilities for reconciliation results
- Custom exceptions for reconciliation errors
"""

from common_lib.data_reconciliation.base import (
    DataReconciliationBase,
    ReconciliationConfig,
    ReconciliationResult,
    ReconciliationStatus,
    ReconciliationSeverity,
    ReconciliationStrategy,
    DataSource,
    DataSourceType,
    Discrepancy,
    DiscrepancyResolution,
)

from common_lib.data_reconciliation.strategies import (
    SourcePriorityStrategy,
    MostRecentStrategy,
    AverageValuesStrategy,
    MedianValuesStrategy,
    CustomResolutionStrategy,
    ThresholdBasedStrategy,
)

from common_lib.data_reconciliation.batch import (
    BatchReconciliationProcessor,
    HistoricalDataReconciliation,
    BulkDataReconciliation,
)

from common_lib.data_reconciliation.realtime import (
    RealTimeReconciliationProcessor,
    StreamingDataReconciliation,
    EventBasedReconciliation,
)

from common_lib.data_reconciliation.exceptions import (
    ReconciliationError,
    SourceDataError,
    ResolutionStrategyError,
    ReconciliationTimeoutError,
    InconsistentDataError,
)

from common_lib.data_reconciliation.reporting import (
    ReconciliationReport,
    ReconciliationMetrics,
    DiscrepancyReport,
    ReconciliationSummary,
)

__all__ = [
    # Base
    'DataReconciliationBase',
    'ReconciliationConfig',
    'ReconciliationResult',
    'ReconciliationStatus',
    'ReconciliationSeverity',
    'ReconciliationStrategy',
    'DataSource',
    'DataSourceType',
    'Discrepancy',
    'DiscrepancyResolution',
    
    # Strategies
    'SourcePriorityStrategy',
    'MostRecentStrategy',
    'AverageValuesStrategy',
    'MedianValuesStrategy',
    'CustomResolutionStrategy',
    'ThresholdBasedStrategy',
    
    # Batch
    'BatchReconciliationProcessor',
    'HistoricalDataReconciliation',
    'BulkDataReconciliation',
    
    # Real-time
    'RealTimeReconciliationProcessor',
    'StreamingDataReconciliation',
    'EventBasedReconciliation',
    
    # Exceptions
    'ReconciliationError',
    'SourceDataError',
    'ResolutionStrategyError',
    'ReconciliationTimeoutError',
    'InconsistentDataError',
    
    # Reporting
    'ReconciliationReport',
    'ReconciliationMetrics',
    'DiscrepancyReport',
    'ReconciliationSummary',
]
