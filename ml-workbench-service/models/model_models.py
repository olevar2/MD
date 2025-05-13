"""
Model Models.

Data models for machine learning models.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Type of machine learning model."""
    
    # Classification models
    LOGISTIC_REGRESSION = "LOGISTIC_REGRESSION"
    RANDOM_FOREST_CLASSIFIER = "RANDOM_FOREST_CLASSIFIER"
    GRADIENT_BOOSTING_CLASSIFIER = "GRADIENT_BOOSTING_CLASSIFIER"
    SVM_CLASSIFIER = "SVM_CLASSIFIER"
    NEURAL_NETWORK_CLASSIFIER = "NEURAL_NETWORK_CLASSIFIER"
    
    # Regression models
    LINEAR_REGRESSION = "LINEAR_REGRESSION"
    RANDOM_FOREST_REGRESSOR = "RANDOM_FOREST_REGRESSOR"
    GRADIENT_BOOSTING_REGRESSOR = "GRADIENT_BOOSTING_REGRESSOR"
    SVM_REGRESSOR = "SVM_REGRESSOR"
    NEURAL_NETWORK_REGRESSOR = "NEURAL_NETWORK_REGRESSOR"
    
    # Time series models
    ARIMA = "ARIMA"
    SARIMA = "SARIMA"
    PROPHET = "PROPHET"
    LSTM = "LSTM"
    GRU = "GRU"
    
    # Ensemble models
    ENSEMBLE = "ENSEMBLE"
    STACKED = "STACKED"
    VOTING = "VOTING"
    
    # Other models
    CUSTOM = "CUSTOM"


class ModelStatus(str, Enum):
    """Status of a model."""
    
    DRAFT = "DRAFT"
    TRAINING = "TRAINING"
    TRAINED = "TRAINED"
    EVALUATING = "EVALUATING"
    EVALUATED = "EVALUATED"
    DEPLOYED = "DEPLOYED"
    ARCHIVED = "ARCHIVED"
    FAILED = "FAILED"


class ModelTrainingStatus(str, Enum):
    """Status of a model training job."""
    
    CREATED = "CREATED"
    PREPARING_DATA = "PREPARING_DATA"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class ModelVersion(BaseModel):
    """Model for a version of a machine learning model."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Model version ID")
    model_id: str = Field(..., description="ID of the model this version belongs to")
    version_number: int = Field(..., description="Version number")
    description: Optional[str] = Field(None, description="Description of the model version")
    status: ModelStatus = Field(default=ModelStatus.DRAFT, description="Status of the model version")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
    experiment_run_id: Optional[str] = Field(None, description="ID of the experiment run that created this model version")
    trained_at: Optional[datetime] = Field(None, description="Time the model was trained")
    trained_by: Optional[str] = Field(None, description="User who trained the model")
    training_job_id: Optional[str] = Field(None, description="ID of the training job that created this model version")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters of the model")
    metrics: Optional[Dict[str, float]] = Field(None, description="Metrics of the model")
    artifact_path: Optional[str] = Field(None, description="Path to the model artifact")
    tags: Optional[Dict[str, str]] = Field(None, description="Tags for the model version")


class Model(BaseModel):
    """Model for a machine learning model."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Model ID")
    name: str = Field(..., description="Model name")
    description: Optional[str] = Field(None, description="Model description")
    model_type: ModelType = Field(..., description="Type of model")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
    created_by: Optional[str] = Field(None, description="User who created the model")
    tags: Optional[Dict[str, str]] = Field(None, description="Tags for the model")
    latest_version_id: Optional[str] = Field(None, description="ID of the latest version of the model")
    latest_version_number: Optional[int] = Field(None, description="Number of the latest version of the model")
    production_version_id: Optional[str] = Field(None, description="ID of the production version of the model")
    versions: Optional[List[ModelVersion]] = Field(None, description="Versions of the model")


class ModelTrainingJob(BaseModel):
    """Model for a model training job."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Job ID")
    model_id: str = Field(..., description="ID of the model being trained")
    model_type: ModelType = Field(..., description="Type of model being trained")
    dataset_id: str = Field(..., description="ID of the dataset used for training")
    experiment_id: Optional[str] = Field(None, description="ID of the experiment to log the training to")
    experiment_run_id: Optional[str] = Field(None, description="ID of the experiment run created for this job")
    model_version_id: Optional[str] = Field(None, description="ID of the model version created by this job")
    status: ModelTrainingStatus = Field(default=ModelTrainingStatus.CREATED, description="Status of the job")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    started_at: Optional[datetime] = Field(None, description="Start time of the job")
    finished_at: Optional[datetime] = Field(None, description="Finish time of the job")
    created_by: Optional[str] = Field(None, description="User who created the job")
    error_message: Optional[str] = Field(None, description="Error message if the job failed")
    
    # Training configuration
    config: Dict[str, Any] = Field(..., description="Training configuration")
    
    # Model parameters and hyperparameters
    parameters: Dict[str, Any] = Field(..., description="Parameters for the model")
    
    # Training parameters
    training_parameters: Dict[str, Any] = Field(..., description="Parameters for the training process")
    
    # Dataset split parameters
    validation_ratio: float = Field(0.2, description="Ratio of data to use for validation")
    test_ratio: float = Field(0.1, description="Ratio of data to use for testing")
    
    # Feature selection
    features: List[str] = Field(..., description="Features used for training")
    target: str = Field(..., description="Target predicted by the model")
    
    # Performance metrics
    training_metrics: Optional[Dict[str, float]] = Field(None, description="Metrics on the training set")
    validation_metrics: Optional[Dict[str, float]] = Field(None, description="Metrics on the validation set")
    test_metrics: Optional[Dict[str, float]] = Field(None, description="Metrics on the test set")
    
    # Tags
    tags: Optional[Dict[str, str]] = Field(None, description="Tags for the job")


class ModelEvaluationResult(BaseModel):
    """Model for model evaluation results."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Evaluation ID")
    model_version_id: str = Field(..., description="ID of the model version evaluated")
    dataset_id: str = Field(..., description="ID of the dataset used for evaluation")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    
    # Metrics
    metrics: Dict[str, float] = Field(..., description="Evaluation metrics")
    
    # Detailed results
    confusion_matrix: Optional[List[List[int]]] = Field(None, description="Confusion matrix for classification models")
    roc_curve: Optional[Dict[str, List[float]]] = Field(None, description="ROC curve data for classification models")
    precision_recall_curve: Optional[Dict[str, List[float]]] = Field(None, description="Precision-recall curve data")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    
    # Other information
    evaluation_parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters used for evaluation")
    notes: Optional[str] = Field(None, description="Notes about the evaluation")