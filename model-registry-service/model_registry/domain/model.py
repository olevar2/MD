"""
Core domain models for the Model Registry Service.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

class ModelType(str, Enum):
    """Type of machine learning model"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    REINFORCEMENT = "reinforcement"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"

class ModelStage(str, Enum):
    """Stage of a model version in its lifecycle"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

class ModelFramework(str, Enum):
    """ML framework used to create the model"""
    SCIKIT_LEARN = "sklearn"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CUSTOM = "custom"

class ModelMetrics(BaseModel):
    """Performance metrics for a model version"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    custom_metrics: Dict[str, float] = {}

class ModelVersion(BaseModel):
    """A specific version of a machine learning model"""
    version_id: str
    model_id: str
    version_number: int
    created_by: str
    created_at: datetime
    updated_at: datetime
    description: Optional[str] = None
    framework: ModelFramework
    framework_version: str
    metrics: Optional[ModelMetrics] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    feature_names: List[str] = []
    target_names: List[str] = []
    artifact_uri: Optional[str] = None
    stage: ModelStage = ModelStage.DEVELOPMENT
    tags: Dict[str, str] = {}
    experiment_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

class ModelMetadata(BaseModel):
    """Metadata for a machine learning model"""
    model_id: str
    name: str
    model_type: ModelType
    description: Optional[str] = None
    created_by: str
    created_at: datetime
    updated_at: datetime
    versions: List[ModelVersion] = []
    latest_version_id: Optional[str] = None
    production_version_id: Optional[str] = None
    staging_version_id: Optional[str] = None
    business_domain: Optional[str] = None
    purpose: Optional[str] = None
    tags: Dict[str, str] = {}
    metadata: Dict[str, Any] = {}
