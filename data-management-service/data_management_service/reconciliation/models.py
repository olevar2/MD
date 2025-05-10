"""
Data Reconciliation Models.

This module defines the data models for the Data Reconciliation system.
It includes models for reconciliation tasks, results, and configurations.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator


class ReconciliationStatus(str, Enum):
    """Status of a reconciliation task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ReconciliationSeverity(str, Enum):
    """Severity of a reconciliation issue."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ReconciliationType(str, Enum):
    """Type of reconciliation."""
    CROSS_SOURCE = "cross_source"  # Compare data from different sources
    TEMPORAL = "temporal"  # Compare data across time
    DERIVED = "derived"  # Compare derived data with source data
    CUSTOM = "custom"  # Custom reconciliation logic


class DataSourceConfig(BaseModel):
    """Configuration for a data source in reconciliation."""
    source_id: str
    source_type: str
    query_params: Dict[str, Any] = Field(default_factory=dict)
    filters: Dict[str, Any] = Field(default_factory=dict)
    transformations: List[Dict[str, Any]] = Field(default_factory=list)


class ReconciliationRule(BaseModel):
    """Rule for data reconciliation."""
    rule_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    field: str
    comparison_type: str  # "exact", "tolerance", "custom"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    severity: ReconciliationSeverity = ReconciliationSeverity.ERROR


class ReconciliationConfig(BaseModel):
    """Configuration for a reconciliation task."""
    config_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    reconciliation_type: ReconciliationType
    primary_source: DataSourceConfig
    secondary_source: Optional[DataSourceConfig] = None
    rules: List[ReconciliationRule] = Field(default_factory=list)
    schedule: Optional[str] = None  # Cron expression for scheduling
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReconciliationIssue(BaseModel):
    """Issue found during reconciliation."""
    issue_id: str = Field(default_factory=lambda: str(uuid4()))
    rule_id: str
    field: str
    primary_value: Any
    secondary_value: Any
    difference: Optional[float] = None
    severity: ReconciliationSeverity
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReconciliationResult(BaseModel):
    """Result of a reconciliation task."""
    result_id: str = Field(default_factory=lambda: str(uuid4()))
    config_id: str
    status: ReconciliationStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    total_records: int = 0
    matched_records: int = 0
    issues: List[ReconciliationIssue] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReconciliationTask(BaseModel):
    """Task for data reconciliation."""
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    config_id: str
    status: ReconciliationStatus = ReconciliationStatus.PENDING
    scheduled_time: datetime
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReconciliationQuery(BaseModel):
    """Query for reconciliation tasks or results."""
    config_id: Optional[str] = None
    status: Optional[ReconciliationStatus] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = 100
    offset: int = 0
