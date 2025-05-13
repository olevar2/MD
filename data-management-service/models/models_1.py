"""
Historical Data Management Models.

This module defines the data models for the Historical Data Management system.
It includes models for historical data records, versions, corrections, and metadata.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator


class DataSourceType(str, Enum):
    """Types of data sources."""
    MARKET = "market"
    TICK = "tick"
    OHLCV = "ohlcv"
    ALTERNATIVE = "alternative"
    ECONOMIC = "economic"
    NEWS = "news"
    SOCIAL = "social"
    CUSTOM = "custom"


class CorrectionType(str, Enum):
    """Types of data corrections."""
    PROVIDER_CORRECTION = "provider_correction"  # Correction from data provider
    MANUAL_CORRECTION = "manual_correction"      # Manual correction by user
    AUTOMATED_CORRECTION = "automated_correction"  # Automated correction by system
    BACKFILL = "backfill"                        # Filling in missing data
    ADJUSTMENT = "adjustment"                    # Adjustment for corporate actions, etc.


class DataQualityIssue(str, Enum):
    """Types of data quality issues."""
    MISSING_DATA = "missing_data"
    OUTLIER = "outlier"
    INCONSISTENT = "inconsistent"
    DUPLICATE = "duplicate"
    STALE = "stale"
    FORMAT_ERROR = "format_error"
    RANGE_ERROR = "range_error"


class HistoricalDataRecord(BaseModel):
    """Base model for historical data records."""
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    timestamp: datetime
    source_type: DataSourceType
    source_id: str
    data: Dict[str, Any]
    version: int = 1
    is_correction: bool = False
    correction_of: Optional[str] = None  # record_id of the record this corrects
    correction_type: Optional[CorrectionType] = None
    correction_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HistoricalOHLCVRecord(HistoricalDataRecord):
    """Model for historical OHLCV data records."""
    timeframe: str
    
    @validator('data')
    def validate_ohlcv_data(cls, v):
        """Validate OHLCV data structure."""
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"OHLCV data must contain {field}")
        return v


class HistoricalTickRecord(HistoricalDataRecord):
    """Model for historical tick data records."""
    
    @validator('data')
    def validate_tick_data(cls, v):
        """Validate tick data structure."""
        required_fields = ['bid', 'ask']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Tick data must contain {field}")
        return v


class HistoricalAlternativeRecord(HistoricalDataRecord):
    """Model for historical alternative data records."""
    data_type: str  # Type of alternative data (e.g., 'news', 'sentiment', 'economic')


class DataCorrectionRecord(BaseModel):
    """Model for tracking data corrections."""
    correction_id: str = Field(default_factory=lambda: str(uuid4()))
    original_record_id: str
    corrected_record_id: str
    correction_type: CorrectionType
    correction_reason: str
    correction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    corrected_by: str
    approved_by: Optional[str] = None
    approval_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataQualityReport(BaseModel):
    """Model for data quality reports."""
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    source_type: DataSourceType
    timeframe: Optional[str] = None
    start_timestamp: datetime
    end_timestamp: datetime
    total_records: int
    missing_records: int
    corrected_records: int
    quality_issues: List[Dict[str, Any]] = Field(default_factory=list)
    report_timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MLDatasetConfig(BaseModel):
    """Configuration for ML dataset generation."""
    dataset_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    symbols: List[str]
    timeframes: List[str]
    start_timestamp: datetime
    end_timestamp: datetime
    features: List[str]
    target: Optional[str] = None
    transformations: List[Dict[str, Any]] = Field(default_factory=list)
    filters: List[Dict[str, Any]] = Field(default_factory=list)
    validation_split: Optional[float] = None
    test_split: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HistoricalDataQuery(BaseModel):
    """Model for querying historical data."""
    symbols: Union[str, List[str]]
    source_type: DataSourceType
    timeframe: Optional[str] = None
    start_timestamp: datetime
    end_timestamp: datetime
    include_corrections: bool = True
    version: Optional[int] = None  # If None, returns latest version
    point_in_time: Optional[datetime] = None  # For point-in-time accurate queries
    filters: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('symbols')
    def validate_symbols(cls, v):
        """Convert single symbol to list."""
        if isinstance(v, str):
            return [v]
        return v
