"""
ML Workbench Service data models.

This module contains Pydantic models for the ML Workbench Service.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union

from pydantic import BaseModel, Field, validator


class ExperimentStatus(str, Enum):
    """Status of an ML experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelFramework(str, Enum):
    """ML framework used for the model."""
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    PROPHET = "prophet"
    STATSMODELS = "statsmodels"
    CUSTOM = "custom"


class DatasetBase(BaseModel):
    """Base class for Dataset models."""
    name: str
    description: Optional[str] = None
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    features: List[str]
    target: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)


class DatasetCreate(DatasetBase):
    """Dataset creation model."""
    pass


class Dataset(DatasetBase):
    """Dataset model."""
    id: str
    created_at: datetime
    updated_at: datetime
    feature_stats: Optional[Dict[str, Any]] = None
    sample_size: int

    class Config:
        """Pydantic config."""
        orm_mode = True


class ExperimentBase(BaseModel):
    """Base class for Experiment models."""
    name: str
    description: Optional[str] = None
    dataset_id: str
    model_type: str
    framework: ModelFramework
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    tags: Optional[List[str]] = Field(default_factory=list)


class ExperimentCreate(ExperimentBase):
    """Experiment creation model."""
    pass


class Experiment(ExperimentBase):
    """Experiment model."""
    id: str
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime
    updated_at: datetime
    metrics: Optional[Dict[str, float]] = Field(default_factory=dict)
    model_artifact_path: Optional[str] = None
    training_duration: Optional[float] = None
    evaluation_results: Optional[Dict[str, Any]] = None
    user_notes: Optional[str] = None
    version: int = 1

    class Config:
        """Pydantic config."""
        orm_mode = True


class ModelBase(BaseModel):
    """Base class for ML Model models."""
    name: str
    description: Optional[str] = None
    experiment_id: str
    version: int = 1
    framework: ModelFramework
    artifact_path: str
    deployed: bool = False
    metrics: Dict[str, float] = Field(default_factory=dict)
    tags: Optional[List[str]] = Field(default_factory=list)


class ModelCreate(ModelBase):
    """Model creation schema."""
    pass


class Model(ModelBase):
    """Model schema."""
    id: str
    created_at: datetime
    updated_at: datetime
    deployment_timestamp: Optional[datetime] = None
    inference_endpoint: Optional[str] = None
    
    class Config:
        """Pydantic config."""
        orm_mode = True


class TrainingConfig(BaseModel):
    """Training configuration model."""
    train_test_split: float = 0.8
    validation_size: float = 0.2
    shuffle: bool = True
    random_state: Optional[int] = None
    stratify_by: Optional[str] = None
    cross_validation_folds: Optional[int] = None
    early_stopping: bool = False
    early_stopping_patience: Optional[int] = None
    max_epochs: Optional[int] = None
    batch_size: Optional[int] = None


class TrainingRequest(BaseModel):
    """Request to start training an experiment."""
    experiment_id: str
    config: TrainingConfig


class EvaluationRequest(BaseModel):
    """Request to evaluate a trained model."""
    model_id: str
    dataset_id: Optional[str] = None
    metrics: List[str]


class PredictionRequest(BaseModel):
    """Request for model prediction."""
    model_id: str
    data: Dict[str, Any]


class BatchPredictionRequest(BaseModel):
    """Request for batch model prediction."""
    model_id: str
    dataset_id: Optional[str] = None
    feature_data: Optional[List[Dict[str, Any]]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class FeatureStat(BaseModel):
    """Feature statistics model."""
    feature_id: str
    feature_name: str
    min: float
    max: float
    mean: float
    median: float
    std: float
    missing_percentage: float
    histogram: Dict[str, List[float]] = Field(
        description="Histogram data with 'bins' and 'values' lists"
    )


class CorrelationMatrix(BaseModel):
    """Correlation matrix model."""
    features: List[str]
    matrix: List[List[float]]
    
    
class FeatureImportance(BaseModel):
    """Feature importance model."""
    features: List[str]
    importance_values: List[float]