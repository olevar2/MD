"""
Dataset Models.

Data models for datasets and feature engineering.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DatasetSource(str, Enum):
    """Source of a dataset."""
    
    FEATURE_STORE = "FEATURE_STORE"
    SQL_QUERY = "SQL_QUERY"
    CSV_FILE = "CSV_FILE"
    PARQUET_FILE = "PARQUET_FILE"
    EXTERNAL_SOURCE = "EXTERNAL_SOURCE"
    DERIVED = "DERIVED"  # Derived from another dataset via feature engineering


class DatasetStatus(str, Enum):
    """Status of a dataset version."""
    
    CREATED = "CREATED"
    PROCESSING = "PROCESSING"
    READY = "READY"
    REFRESHING = "REFRESHING"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


class FeatureTransformType(str, Enum):
    """Type of feature transformation."""
    
    NORMALIZE = "NORMALIZE"
    STANDARDIZE = "STANDARDIZE"
    LOG_TRANSFORM = "LOG_TRANSFORM"
    ONE_HOT_ENCODE = "ONE_HOT_ENCODE"
    LABEL_ENCODE = "LABEL_ENCODE"
    BIN = "BIN"
    CUSTOM = "CUSTOM"


class FeatureEngineering(BaseModel):
    """Model for a feature engineering operation."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Operation ID")
    dataset_version_id: str = Field(..., description="ID of the dataset version")
    name: str = Field(..., description="Name of the operation")
    type: str = Field(..., description="Type of operation")
    parameters: Dict[str, Any] = Field(..., description="Parameters for the operation")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    created_by: Optional[str] = Field(None, description="User who created the operation")
    input_columns: List[str] = Field(..., description="Input columns for the operation")
    output_columns: List[str] = Field(..., description="Output columns produced by the operation")
    description: Optional[str] = Field(None, description="Description of the operation")


class FeatureDefinition(BaseModel):
    """Model for a feature definition in a dataset."""
    
    name: str = Field(..., description="Name of the feature")
    feature_id: str = Field(..., description="ID of the feature in the feature store")
    data_type: str = Field(..., description="Data type of the feature")
    is_target: bool = Field(False, description="Whether this feature is a prediction target")
    description: Optional[str] = Field(None, description="Description of the feature")
    transformation: Optional[str] = Field(None, description="Transformation applied to the feature")
    statistics: Optional[Dict[str, Any]] = Field(None, description="Statistics of the feature")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the feature")


class DatasetVersion(BaseModel):
    """Model for a version of a dataset."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Dataset version ID")
    dataset_id: str = Field(..., description="ID of the dataset this version belongs to")
    version_number: int = Field(..., description="Version number")
    description: Optional[str] = Field(None, description="Description of the dataset version")
    status: DatasetStatus = Field(default=DatasetStatus.CREATED, description="Status of the dataset version")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
    created_by: Optional[str] = Field(None, description="User who created the dataset version")
    
    # Source information
    source: DatasetSource = Field(..., description="Source of the dataset")
    source_details: Optional[Dict[str, Any]] = Field(None, description="Details about the source")
    parent_version_id: Optional[str] = Field(None, description="ID of the parent version (for derived datasets)")
    
    # Data characteristics
    features: List[FeatureDefinition] = Field(..., description="Features in the dataset")
    row_count: Optional[int] = Field(None, description="Number of rows in the dataset")
    start_time: Optional[datetime] = Field(None, description="Start time of the data in the dataset")
    end_time: Optional[datetime] = Field(None, description="End time of the data in the dataset")
    
    # Feature engineering
    feature_engineering_operations: Optional[List[FeatureEngineering]] = Field(None, 
                                                                              description="Feature engineering operations applied")
    
    # Additional information
    tags: Optional[Dict[str, str]] = Field(None, description="Tags for the dataset version")
    symbols: Optional[List[str]] = Field(None, description="Symbols included in the dataset")
    timeframes: Optional[List[str]] = Field(None, description="Timeframes included in the dataset")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the dataset version")
    storage_path: Optional[str] = Field(None, description="Path where the dataset is stored")


class Dataset(BaseModel):
    """Model for a dataset."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Dataset ID")
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
    created_by: Optional[str] = Field(None, description="User who created the dataset")
    tags: Optional[Dict[str, str]] = Field(None, description="Tags for the dataset")
    latest_version_id: Optional[str] = Field(None, description="ID of the latest version of the dataset")
    latest_version_number: Optional[int] = Field(None, description="Number of the latest version of the dataset")
    versions: Optional[List[DatasetVersion]] = Field(None, description="Versions of the dataset")


class DatasetPreview(BaseModel):
    """Model for a preview of a dataset."""
    
    dataset_id: str = Field(..., description="ID of the dataset")
    dataset_version_id: str = Field(..., description="ID of the dataset version")
    columns: List[str] = Field(..., description="Columns in the preview")
    data: List[Dict[str, Any]] = Field(..., description="Preview data")
    total_rows: int = Field(..., description="Total number of rows in the dataset")
    preview_rows: int = Field(..., description="Number of rows in the preview")


class DatasetStatistics(BaseModel):
    """Model for statistics of a dataset."""
    
    dataset_id: str = Field(..., description="ID of the dataset")
    dataset_version_id: str = Field(..., description="ID of the dataset version")
    row_count: int = Field(..., description="Number of rows in the dataset")
    column_stats: Dict[str, Dict[str, Any]] = Field(..., description="Statistics for each column")
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Correlation matrix")
    histograms: Optional[Dict[str, Dict[str, List[Any]]]] = Field(None, description="Histogram data for numeric columns")
    created_at: datetime = Field(default_factory=datetime.now, description="Time the statistics were calculated")


class DatasetSchema(BaseModel):
    """Model for the schema of a dataset."""
    
    dataset_id: str = Field(..., description="ID of the dataset")
    dataset_version_id: str = Field(..., description="ID of the dataset version")
    columns: Dict[str, Dict[str, Any]] = Field(..., description="Schema of each column")
    primary_key: Optional[Union[str, List[str]]] = Field(None, description="Primary key column(s)")
    time_column: Optional[str] = Field(None, description="Column representing time")