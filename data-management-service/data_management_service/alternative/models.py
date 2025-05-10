"""
Alternative Data Models.

This module defines the core domain models for the Alternative Data Integration framework.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


class AlternativeDataType(str, Enum):
    """Types of alternative data supported by the framework."""
    NEWS = "news"
    ECONOMIC = "economic"
    SENTIMENT = "sentiment"
    SOCIAL_MEDIA = "social_media"
    WEATHER = "weather"
    SATELLITE = "satellite"
    CORPORATE_EVENTS = "corporate_events"
    REGULATORY = "regulatory"
    CUSTOM = "custom"


class DataFrequency(str, Enum):
    """Frequency of data updates."""
    REAL_TIME = "real_time"
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    IRREGULAR = "irregular"


class DataReliability(str, Enum):
    """Reliability level of the data source."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    EXPERIMENTAL = "experimental"


class DataSourceMetadata(BaseModel):
    """Metadata about a data source."""
    name: str
    description: str
    provider: str
    frequency: DataFrequency
    reliability: DataReliability
    last_updated: datetime
    coverage_start: Optional[datetime] = None
    coverage_end: Optional[datetime] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class AlternativeDataSource(BaseModel):
    """Configuration for an alternative data source."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: AlternativeDataType
    description: str
    metadata: DataSourceMetadata
    config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DataValidationRule(BaseModel):
    """Rule for validating alternative data."""
    field: str
    rule_type: str  # e.g., "required", "format", "range", "enum"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    error_message: str
    severity: str = "error"  # "error", "warning", "info"


class DataTransformationRule(BaseModel):
    """Rule for transforming alternative data."""
    field: str
    transformation_type: str  # e.g., "format", "normalize", "aggregate"
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AlternativeDataSchema(BaseModel):
    """Schema definition for an alternative data type."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    data_type: AlternativeDataType
    description: str
    version: str = "1.0.0"
    fields: Dict[str, Dict[str, Any]]
    validation_rules: List[DataValidationRule] = Field(default_factory=list)
    transformation_rules: List[DataTransformationRule] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CorrelationMetric(BaseModel):
    """Correlation metric between alternative data and market movements."""
    instrument: str
    timeframe: str
    correlation_value: float
    p_value: float
    sample_size: int
    start_date: datetime
    end_date: datetime
    methodology: str
    notes: Optional[str] = None


class AlternativeDataCorrelation(BaseModel):
    """Correlation analysis between alternative data and market movements."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    data_type: AlternativeDataType
    source_id: str
    metrics: List[CorrelationMetric]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeatureExtractionConfig(BaseModel):
    """Configuration for feature extraction from alternative data."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    data_type: AlternativeDataType
    description: str
    extraction_method: str  # e.g., "nlp", "statistical", "custom"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    output_features: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
