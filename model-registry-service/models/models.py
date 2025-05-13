"""
API models (request/response schemas) for the Model Registry Service.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from models.model import (
    ModelType,
    ModelStage,
    ModelFramework,
    ModelMetadata,
    ModelVersion,
    ModelMetrics
)

# Request Models

class ModelCreate(BaseModel):
    """Request model for creating a new model"""
    name: str = Field(..., description="Name of the model")
    model_type: ModelType = Field(..., description="Type of the model")
    description: Optional[str] = Field(None, description="Description of the model")
    created_by: str = Field("system", description="User or system that created the model")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags for the model")
    business_domain: Optional[str] = Field(None, description="Business domain the model belongs to")
    purpose: Optional[str] = Field(None, description="Purpose of the model")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ModelUpdate(BaseModel):
    """Request model for updating a model"""
    description: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    business_domain: Optional[str] = None
    purpose: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class VersionCreate(BaseModel):
    """Request model for creating a new version"""
    framework: ModelFramework = Field(..., description="ML framework used")
    framework_version: str = Field(..., description="Version of the ML framework")
    created_by: str = Field("system", description="User or system that created the version")
    description: Optional[str] = Field(None, description="Description of the version")
    metrics: Optional[Dict[str, float]] = Field(None, description="Performance metrics")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Model hyperparameters")
    feature_names: List[str] = Field(default_factory=list, description="Names of input features")
    target_names: List[str] = Field(default_factory=list, description="Names of target variables")
    experiment_id: Optional[str] = Field(None, description="ID of associated experiment")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags for the version")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    stage: ModelStage = Field(default=ModelStage.DEVELOPMENT, description="Initial stage")

class StageUpdate(BaseModel):
    """Request model for updating a version's stage"""
    stage: ModelStage = Field(..., description="New stage for the version")

# Response Models

class MetricsResponse(BaseModel):
    """Response model for model metrics"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    custom_metrics: Dict[str, float] = {}

    @classmethod
    def from_domain(cls, metrics: ModelMetrics) -> "MetricsResponse":
        return cls(**metrics.dict())

class VersionResponse(BaseModel):
    """Response model for model version details"""
    version_id: str
    model_id: str
    version_number: int
    created_by: str
    created_at: datetime
    updated_at: datetime
    description: Optional[str] = None
    framework: ModelFramework
    framework_version: str
    metrics: Optional[MetricsResponse] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    feature_names: List[str]
    target_names: List[str]
    artifact_uri: Optional[str] = None
    stage: ModelStage
    tags: Dict[str, str]
    experiment_id: Optional[str] = None
    metadata: Dict[str, Any]

    @classmethod
    def from_domain(cls, version: ModelVersion) -> "VersionResponse":
    """
    From domain.
    
    Args:
        version: Description of version
    
    Returns:
        "VersionResponse": Description of return value
    
    """

        data = version.dict()
        if data.get("metrics"):
            data["metrics"] = MetricsResponse.from_domain(version.metrics)
        return cls(**data)

class ModelResponse(BaseModel):
    """Response model for model details"""
    model_id: str
    name: str
    model_type: ModelType
    description: Optional[str] = None
    created_by: str
    created_at: datetime
    updated_at: datetime
    versions: List[VersionResponse]
    latest_version_id: Optional[str] = None
    production_version_id: Optional[str] = None
    staging_version_id: Optional[str] = None
    business_domain: Optional[str] = None
    purpose: Optional[str] = None
    tags: Dict[str, str]
    metadata: Dict[str, Any]

    @classmethod
    def from_domain(cls, model: ModelMetadata) -> "ModelResponse":
        data = model.dict()
        data["versions"] = [VersionResponse.from_domain(v) for v in model.versions]
        return cls(**data)

class ModelsResponse(BaseModel):
    """Response model for listing models"""
    models: List[ModelResponse]

class VersionsResponse(BaseModel):
    """Response model for listing versions"""
    versions: List[VersionResponse]

class ErrorResponse(BaseModel):
    """Response model for errors"""
    detail: str
