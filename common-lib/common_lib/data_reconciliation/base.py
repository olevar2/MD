"""
Base classes and interfaces for data reconciliation.

This module provides the foundation for the data reconciliation framework,
including base classes, enums, and data structures.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

import pandas as pd
import numpy as np

from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class DataSourceType(Enum):
    """Types of data sources for reconciliation."""
    DATABASE = auto()
    API = auto()
    FILE = auto()
    STREAM = auto()
    CACHE = auto()
    EVENT = auto()
    MODEL = auto()
    CUSTOM = auto()


class ReconciliationStatus(Enum):
    """Status of a reconciliation process."""
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    PARTIALLY_COMPLETED = auto()
    TIMEOUT = auto()


class ReconciliationSeverity(Enum):
    """Severity levels for reconciliation discrepancies."""
    CRITICAL = auto()  # Requires immediate attention
    HIGH = auto()      # Significant discrepancy
    MEDIUM = auto()    # Notable discrepancy
    LOW = auto()       # Minor discrepancy
    INFO = auto()      # Informational only


class ReconciliationStrategy(Enum):
    """Strategies for resolving discrepancies."""
    SOURCE_PRIORITY = auto()  # Use data from the highest priority source
    MOST_RECENT = auto()      # Use the most recently updated data
    AVERAGE = auto()          # Use average of all values
    MEDIAN = auto()           # Use median of all values
    CUSTOM = auto()           # Use a custom resolution function
    THRESHOLD_BASED = auto()  # Use threshold to determine resolution method


class DataSource:
    """Represents a data source for reconciliation."""
    
    def __init__(
        self,
        source_id: str,
        name: str,
        source_type: DataSourceType,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a data source.
        
        Args:
            source_id: Unique identifier for the source
            name: Human-readable name of the source
            source_type: Type of the data source
            priority: Priority of the source (higher number = higher priority)
            metadata: Additional metadata about the source
        """
        self.source_id = source_id
        self.name = name
        self.source_type = source_type
        self.priority = priority
        self.metadata = metadata or {}
        
    def __repr__(self) -> str:
        return f"DataSource(id={self.source_id}, name={self.name}, type={self.source_type.name}, priority={self.priority})"


class Discrepancy:
    """Represents a discrepancy between data sources."""
    
    def __init__(
        self,
        discrepancy_id: Optional[str] = None,
        field: str = "",
        sources: Optional[Dict[str, Any]] = None,
        severity: ReconciliationSeverity = ReconciliationSeverity.MEDIUM,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a discrepancy.
        
        Args:
            discrepancy_id: Unique identifier for the discrepancy
            field: Field or attribute where the discrepancy was found
            sources: Dictionary mapping source IDs to their values
            severity: Severity level of the discrepancy
            timestamp: When the discrepancy was detected
            metadata: Additional metadata about the discrepancy
        """
        self.discrepancy_id = discrepancy_id or str(uuid.uuid4())
        self.field = field
        self.sources = sources or {}
        self.severity = severity
        self.timestamp = timestamp or datetime.utcnow()
        self.metadata = metadata or {}
        
        # Calculate statistics
        self._calculate_statistics()
        
    def _calculate_statistics(self) -> None:
        """Calculate statistics about the discrepancy."""
        values = list(self.sources.values())
        if not values or not all(isinstance(v, (int, float)) for v in values):
            self.min_value = None
            self.max_value = None
            self.mean_value = None
            self.median_value = None
            self.std_dev = None
            self.range_pct = None
            return
            
        self.min_value = min(values)
        self.max_value = max(values)
        self.mean_value = sum(values) / len(values)
        self.median_value = sorted(values)[len(values) // 2]
        
        if len(values) > 1:
            self.std_dev = np.std(values)
        else:
            self.std_dev = 0
            
        if self.mean_value != 0:
            self.range_pct = (self.max_value - self.min_value) / abs(self.mean_value) * 100
        else:
            self.range_pct = None
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "discrepancy_id": self.discrepancy_id,
            "field": self.field,
            "sources": self.sources,
            "severity": self.severity.name,
            "timestamp": self.timestamp.isoformat(),
            "statistics": {
                "min_value": self.min_value,
                "max_value": self.max_value,
                "mean_value": self.mean_value,
                "median_value": self.median_value,
                "std_dev": self.std_dev,
                "range_pct": self.range_pct
            },
            "metadata": self.metadata
        }
        
    def __repr__(self) -> str:
        return f"Discrepancy(id={self.discrepancy_id}, field={self.field}, severity={self.severity.name})"


class DiscrepancyResolution:
    """Represents the resolution of a discrepancy."""
    
    def __init__(
        self,
        discrepancy: Discrepancy,
        resolved_value: Any,
        strategy: ReconciliationStrategy,
        resolution_source: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a discrepancy resolution.
        
        Args:
            discrepancy: The discrepancy being resolved
            resolved_value: The value used to resolve the discrepancy
            strategy: Strategy used to resolve the discrepancy
            resolution_source: Source of the resolved value (if applicable)
            timestamp: When the resolution occurred
            metadata: Additional metadata about the resolution
        """
        self.discrepancy = discrepancy
        self.resolved_value = resolved_value
        self.strategy = strategy
        self.resolution_source = resolution_source
        self.timestamp = timestamp or datetime.utcnow()
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "discrepancy_id": self.discrepancy.discrepancy_id,
            "field": self.discrepancy.field,
            "resolved_value": self.resolved_value,
            "strategy": self.strategy.name,
            "resolution_source": self.resolution_source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
        
    def __repr__(self) -> str:
        return f"DiscrepancyResolution(discrepancy_id={self.discrepancy.discrepancy_id}, strategy={self.strategy.name})"


class ReconciliationConfig:
    """Configuration for a reconciliation process."""
    
    def __init__(
        self,
        sources: List[DataSource],
        strategy: ReconciliationStrategy = ReconciliationStrategy.SOURCE_PRIORITY,
        tolerance: float = 0.0001,
        fields_to_reconcile: Optional[List[str]] = None,
        fields_to_ignore: Optional[List[str]] = None,
        timeout_seconds: Optional[float] = None,
        batch_size: int = 1000,
        auto_resolve: bool = False,
        notification_threshold: ReconciliationSeverity = ReconciliationSeverity.HIGH,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize reconciliation configuration.
        
        Args:
            sources: List of data sources to reconcile
            strategy: Strategy for resolving discrepancies
            tolerance: Tolerance for numerical differences
            fields_to_reconcile: Specific fields to reconcile (if None, reconcile all)
            fields_to_ignore: Fields to exclude from reconciliation
            timeout_seconds: Maximum time for reconciliation process
            batch_size: Size of batches for processing large datasets
            auto_resolve: Whether to automatically resolve discrepancies
            notification_threshold: Minimum severity for notifications
            metadata: Additional configuration metadata
        """
        self.sources = sources
        self.strategy = strategy
        self.tolerance = tolerance
        self.fields_to_reconcile = fields_to_reconcile
        self.fields_to_ignore = fields_to_ignore or []
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size
        self.auto_resolve = auto_resolve
        self.notification_threshold = notification_threshold
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "sources": [s.source_id for s in self.sources],
            "strategy": self.strategy.name,
            "tolerance": self.tolerance,
            "fields_to_reconcile": self.fields_to_reconcile,
            "fields_to_ignore": self.fields_to_ignore,
            "timeout_seconds": self.timeout_seconds,
            "batch_size": self.batch_size,
            "auto_resolve": self.auto_resolve,
            "notification_threshold": self.notification_threshold.name,
            "metadata": self.metadata
        }


class ReconciliationResult:
    """Results of a reconciliation process."""
    
    def __init__(
        self,
        reconciliation_id: str,
        config: ReconciliationConfig,
        status: ReconciliationStatus = ReconciliationStatus.NOT_STARTED,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        discrepancies: Optional[List[Discrepancy]] = None,
        resolutions: Optional[List[DiscrepancyResolution]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize reconciliation result.
        
        Args:
            reconciliation_id: Unique identifier for the reconciliation
            config: Configuration used for the reconciliation
            status: Status of the reconciliation process
            start_time: When the reconciliation started
            end_time: When the reconciliation completed
            discrepancies: List of detected discrepancies
            resolutions: List of discrepancy resolutions
            metadata: Additional result metadata
        """
        self.reconciliation_id = reconciliation_id
        self.config = config
        self.status = status
        self.start_time = start_time or datetime.utcnow()
        self.end_time = end_time
        self.discrepancies = discrepancies or []
        self.resolutions = resolutions or []
        self.metadata = metadata or {}
        
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get the duration of the reconciliation in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
        
    @property
    def discrepancy_count(self) -> int:
        """Get the total number of discrepancies."""
        return len(self.discrepancies)
        
    @property
    def resolution_count(self) -> int:
        """Get the total number of resolutions."""
        return len(self.resolutions)
        
    @property
    def resolution_rate(self) -> float:
        """Get the percentage of discrepancies that were resolved."""
        if self.discrepancy_count == 0:
            return 100.0
        return (self.resolution_count / self.discrepancy_count) * 100
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "reconciliation_id": self.reconciliation_id,
            "config": self.config.to_dict(),
            "status": self.status.name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "discrepancy_count": self.discrepancy_count,
            "resolution_count": self.resolution_count,
            "resolution_rate": self.resolution_rate,
            "discrepancies": [d.to_dict() for d in self.discrepancies],
            "resolutions": [r.to_dict() for r in self.resolutions],
            "metadata": self.metadata
        }
        
    def __repr__(self) -> str:
        return f"ReconciliationResult(id={self.reconciliation_id}, status={self.status.name}, discrepancies={self.discrepancy_count}, resolutions={self.resolution_count})"


class DataReconciliationBase(ABC):
    """Base class for data reconciliation implementations."""
    
    def __init__(self, config: ReconciliationConfig):
        """
        Initialize data reconciliation.
        
        Args:
            config: Configuration for the reconciliation process
        """
        self.config = config
        self.reconciliation_id = str(uuid.uuid4())
        self.result = ReconciliationResult(
            reconciliation_id=self.reconciliation_id,
            config=config,
            status=ReconciliationStatus.NOT_STARTED
        )
        
    @abstractmethod
    async def fetch_data(self, source: DataSource, **kwargs) -> Any:
        """
        Fetch data from a source.
        
        Args:
            source: The data source to fetch from
            **kwargs: Additional parameters for fetching
            
        Returns:
            The fetched data
        """
        pass
        
    @abstractmethod
    async def compare_data(self, data_map: Dict[str, Any]) -> List[Discrepancy]:
        """
        Compare data from different sources to identify discrepancies.
        
        Args:
            data_map: Dictionary mapping source IDs to their data
            
        Returns:
            List of identified discrepancies
        """
        pass
        
    @abstractmethod
    async def resolve_discrepancies(self, discrepancies: List[Discrepancy]) -> List[DiscrepancyResolution]:
        """
        Resolve discrepancies using the configured strategy.
        
        Args:
            discrepancies: List of discrepancies to resolve
            
        Returns:
            List of discrepancy resolutions
        """
        pass
        
    @abstractmethod
    async def apply_resolutions(self, resolutions: List[DiscrepancyResolution]) -> bool:
        """
        Apply resolutions to the data.
        
        Args:
            resolutions: List of resolutions to apply
            
        Returns:
            Whether all resolutions were successfully applied
        """
        pass
        
    async def reconcile(self, **kwargs) -> ReconciliationResult:
        """
        Perform the reconciliation process.
        
        Args:
            **kwargs: Additional parameters for reconciliation
            
        Returns:
            Results of the reconciliation process
        """
        self.result.status = ReconciliationStatus.IN_PROGRESS
        self.result.start_time = datetime.utcnow()
        
        try:
            # Step 1: Fetch data from all sources
            data_map = {}
            for source in self.config.sources:
                data_map[source.source_id] = await self.fetch_data(source, **kwargs)
                
            # Step 2: Compare data to identify discrepancies
            discrepancies = await self.compare_data(data_map)
            self.result.discrepancies = discrepancies
            
            # Step 3: Resolve discrepancies if auto-resolve is enabled
            if self.config.auto_resolve and discrepancies:
                resolutions = await self.resolve_discrepancies(discrepancies)
                self.result.resolutions = resolutions
                
                # Step 4: Apply resolutions
                success = await self.apply_resolutions(resolutions)
                if not success:
                    self.result.status = ReconciliationStatus.PARTIALLY_COMPLETED
                else:
                    self.result.status = ReconciliationStatus.COMPLETED
            else:
                self.result.status = ReconciliationStatus.COMPLETED
                
        except Exception as e:
            logger.error(f"Reconciliation failed: {str(e)}")
            self.result.status = ReconciliationStatus.FAILED
            self.result.metadata["error"] = str(e)
            
        finally:
            self.result.end_time = datetime.utcnow()
            
        return self.result
