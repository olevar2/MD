"""
Experiment Models Module.

Contains data models for ML workbench experiments, datasets, models and predictions.
"""

from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Any, TypedDict
from pydantic import BaseModel, Field


class ExperimentStatus(str, Enum):
    """Status of an experiment or training run."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelType(str, Enum):
    """Type of machine learning model."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class ModelFramework(str, Enum):
    """ML framework used for the model."""
    SCIKIT_LEARN = "scikit_learn"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    PROPHET = "prophet"
    CUSTOM = "custom"


class TrainingAlgorithm(str, Enum):
    """Algorithm used for training."""
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVM = "svm"
    KNN = "knn"
    NAIVE_BAYES = "naive_bayes"
    NEURAL_NETWORK = "neural_network"
    CNN = "cnn"
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL_CLUSTERING = "hierarchical_clustering"
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    ARIMA = "arima"
    SARIMA = "sarima"
    PROPHET = "prophet"
    CUSTOM = "custom"


class MetricValue(BaseModel):
    """A single metric value with metadata."""
    name: str
    value: Union[float, Dict[str, float]]  # Support both scalar and dictionary metrics
    higher_is_better: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Dataset(BaseModel):
    """A dataset created from the feature store."""
    id: str
    name: str
    description: Optional[str] = None
    features: List[str]  # List of feature IDs
    target: Optional[str] = None  # Target feature ID
    size: int  # Number of samples
    start_date: datetime
    end_date: datetime
    symbols: List[str]
    timeframes: List[str]
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None


class Experiment(BaseModel):
    """An ML experiment."""
    id: str
    name: str
    description: Optional[str] = None
    status: ExperimentStatus
    model_type: ModelType
    model_framework: ModelFramework
    algorithm: TrainingAlgorithm
    dataset_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)  # Metrics for the best run
    artifacts: Dict[str, str] = Field(default_factory=dict)  # Mapping of artifact name to path
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: Optional[str] = None
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TrainingRun(BaseModel):
    """A single training run within an experiment."""
    id: str
    experiment_id: str
    run_number: int
    status: ExperimentStatus
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, str] = Field(default_factory=dict)  # Mapping of artifact name to path
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Model(BaseModel):
    """A trained ML model."""
    id: str
    name: str
    version: str
    description: Optional[str] = None
    experiment_id: str
    training_run_id: Optional[str] = None
    model_type: ModelType
    model_framework: ModelFramework
    algorithm: TrainingAlgorithm
    parameters: Dict[str, Any]
    metrics: Dict[str, Any]
    artifacts: Dict[str, str]  # Mapping of artifact name to path
    features: List[str]  # List of input feature names
    target: Optional[str] = None  # Target variable name
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    is_published: bool = False
    tags: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class Prediction(BaseModel):
    """A record of a prediction made by a model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    input_data: Dict[str, Any]
    prediction: Any
    confidence: Optional[float] = None
    actual_value: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    additional_outputs: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None


class ExperimentFilter(BaseModel):
    """Filter criteria for experiments."""
    name: Optional[str] = None
    status: Optional[List[ExperimentStatus]] = None
    model_type: Optional[List[ModelType]] = None
    model_framework: Optional[List[ModelFramework]] = None
    algorithm: Optional[List[TrainingAlgorithm]] = None
    dataset_id: Optional[str] = None
    tags: Optional[List[str]] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    created_by: Optional[str] = None