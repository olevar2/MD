"""
Experiment Service Module.

Provides core services for managing ML experiments and datasets.
"""
import os
import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from core_foundations.utils.logger import get_logger
from ml_workbench_service.clients.feature_store_client import FeatureStoreClient
from ml_workbench_service.models.experiment_models import (
    Dataset, Experiment, TrainingRun, Model, Prediction,
    ExperimentStatus, ModelType, ModelFramework, TrainingAlgorithm, ExperimentFilter,
    MetricValue
)
from ml_workbench_service.repositories.experiment_repository import ExperimentRepository

logger = get_logger("experiment-service")


class ExperimentService:
    """
    Service for managing machine learning experiments and models.
    
    This service provides methods for:
    - Creating and managing datasets from feature store data
    - Running machine learning experiments
    - Evaluating and comparing models
    - Storing and retrieving experiment results
    """
    
    def __init__(
        self, 
        experiment_repository: ExperimentRepository,
        feature_store_client: FeatureStoreClient,
        model_storage_path: str
    ):
        """
        Initialize the experiment service.
        
        Args:
            experiment_repository: Repository for experiment data
            feature_store_client: Client for the feature store
            model_storage_path: Base path for storing model artifacts
        """
        self.repository = experiment_repository
        self.feature_store = feature_store_client
        self.model_storage_path = model_storage_path
        
        # Ensure the model storage path exists
        os.makedirs(model_storage_path, exist_ok=True)
        
    # Dataset methods
    
    async def create_dataset_from_feature_store(
        self,
        name: str,
        description: Optional[str],
        symbols: List[str],
        timeframes: List[str],
        feature_ids: List[str],
        target_feature: Optional[str],
        start_date: datetime,
        end_date: datetime,
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dataset:
        """
        Create a new dataset from feature store data.
        
        Args:
            name: Name for the dataset
            description: Description of the dataset
            symbols: List of symbols to include
            timeframes: List of timeframes to include
            feature_ids: List of feature IDs to include
            target_feature: Name of the target feature (optional)
            start_date: Start date for the data
            end_date: End date for the data
            train_ratio: Proportion of data for training
            validation_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            metadata: Additional metadata
            
        Returns:
            Dataset object describing the created dataset
        """
        try:
            # Validate the ratios
            if train_ratio + validation_ratio + test_ratio != 1.0:
                raise ValueError("Train, validation, and test ratios must sum to 1.0")
            
            # Generate dataset ID
            dataset_id = str(uuid.uuid4())
            
            # Create storage directory
            dataset_path = os.path.join(self.model_storage_path, "datasets", dataset_id)
            os.makedirs(dataset_path, exist_ok=True)
            
            # Initialize counters for total size
            total_samples = 0
            
            # Initialize metadata if not provided
            if metadata is None:
                metadata = {}
                
            # Add storage location to metadata
            metadata["storage_path"] = dataset_path
            
            # Fetch data for each symbol and timeframe
            all_features_data = []
            for symbol in symbols:
                for timeframe in timeframes:
                    logger.info(f"Fetching feature data for {symbol}/{timeframe}")
                    
                    try:
                        # Fetch feature vectors
                        feature_data = await self.feature_store.get_feature_vectors(
                            feature_ids=feature_ids,
                            symbol=symbol,
                            timeframe=timeframe,
                            start_date=start_date,
                            end_date=end_date
                        )
                        
                        # Add symbol and timeframe columns
                        feature_data['symbol'] = symbol
                        feature_data['timeframe'] = timeframe
                        
                        # Append to the list
                        all_features_data.append(feature_data)
                        
                    except Exception as e:
                        logger.warning(
                            f"Error fetching features for {symbol}/{timeframe}: {str(e)}"
                        )
                        
            # Combine all fetched data
            if not all_features_data:
                raise ValueError("No feature data was successfully fetched")
                
            combined_data = pd.concat(all_features_data, axis=0)
            
            # Reset index to make timestamp a column
            combined_data.reset_index(inplace=True)
            
            # Record the total size
            total_samples = len(combined_data)
            
            # Save the combined data
            combined_data_path = os.path.join(dataset_path, "combined_data.csv")
            combined_data.to_csv(combined_data_path, index=False)
            
            # Create train/validation/test splits
            # We'll split chronologically for time series data
            combined_data = combined_data.sort_values(by='timestamp')
            
            # Calculate split indices
            train_end_idx = int(total_samples * train_ratio)
            val_end_idx = int(total_samples * (train_ratio + validation_ratio))
            
            # Split the data
            train_data = combined_data.iloc[:train_end_idx]
            val_data = combined_data.iloc[train_end_idx:val_end_idx]
            test_data = combined_data.iloc[val_end_idx:]
            
            # Save the splits
            train_path = os.path.join(dataset_path, "train_data.csv")
            val_path = os.path.join(dataset_path, "validation_data.csv")
            test_path = os.path.join(dataset_path, "test_data.csv")
            
            train_data.to_csv(train_path, index=False)
            val_data.to_csv(val_path, index=False)
            test_data.to_csv(test_path, index=False)
            
            # Add split info to metadata
            metadata["train_samples"] = len(train_data)
            metadata["validation_samples"] = len(val_data)
            metadata["test_samples"] = len(test_data)
            metadata["data_paths"] = {
                "combined": combined_data_path,
                "train": train_path,
                "validation": val_path,
                "test": test_path
            }
            
            # Create the dataset object
            dataset = Dataset(
                id=dataset_id,
                name=name,
                description=description,
                features=feature_ids,
                target=target_feature,
                size=total_samples,
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
                timeframes=timeframes,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
                created_at=datetime.utcnow(),
                metadata=metadata
            )
            
            # Save to repository
            await self.repository.create_dataset(dataset)
            
            logger.info(
                f"Created dataset {dataset_id} with {total_samples} samples "
                f"across {len(symbols)} symbols and {len(timeframes)} timeframes"
            )
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            raise
    
    async def get_dataset(self, dataset_id: str) -> Dataset:
        """
        Get a dataset by ID.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Dataset object
            
        Raises:
            ValueError: If the dataset is not found
        """
        dataset = await self.repository.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        return dataset
    
    async def list_datasets(
        self,
        name_filter: Optional[str] = None,
        symbol_filter: Optional[str] = None,
        timeframe_filter: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dataset]:
        """
        List datasets with optional filtering.
        
        Args:
            name_filter: Filter by name (partial match)
            symbol_filter: Filter by symbol (exact match)
            timeframe_filter: Filter by timeframe (exact match)
            limit: Maximum number of datasets to return
            offset: Number of datasets to skip
            
        Returns:
            List of datasets matching the criteria
        """
        return await self.repository.list_datasets(
            limit=limit,
            offset=offset,
            name_filter=name_filter,
            symbol_filter=symbol_filter,
            timeframe_filter=timeframe_filter
        )
        
    async def load_dataset_data(
        self,
        dataset_id: str,
        subset: str = "combined"
    ) -> pd.DataFrame:
        """
        Load dataset data from storage.
        
        Args:
            dataset_id: ID of the dataset
            subset: Which data subset to load ("combined", "train", "validation", "test")
            
        Returns:
            DataFrame containing the dataset data
            
        Raises:
            ValueError: If the dataset is not found or the subset is invalid
        """
        # Get dataset metadata
        dataset = await self.get_dataset(dataset_id)
        
        # Get data path from metadata
        if not dataset.metadata or "data_paths" not in dataset.metadata:
            raise ValueError(f"Dataset {dataset_id} does not have valid file paths")
            
        data_paths = dataset.metadata["data_paths"]
        
        if subset not in data_paths:
            raise ValueError(f"Invalid subset '{subset}'. Must be one of: {list(data_paths.keys())}")
            
        file_path = data_paths[subset]
        
        # Load the data
        if not os.path.exists(file_path):
            raise ValueError(f"Dataset file not found: {file_path}")
            
        return pd.read_csv(file_path)
    
    # Experiment methods
    
    async def create_experiment(
        self,
        name: str,
        description: Optional[str],
        dataset_id: str,
        model_type: ModelType,
        model_framework: ModelFramework,
        algorithm: TrainingAlgorithm,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Experiment:
        """
        Create a new experiment.
        
        Args:
            name: Name for the experiment
            description: Description of the experiment
            dataset_id: ID of the dataset to use
            model_type: Type of model (classification, regression, etc.)
            model_framework: Framework to use (scikit-learn, tensorflow, etc.)
            algorithm: Algorithm to use
            parameters: Training parameters
            tags: Tags for organization
            created_by: User who created the experiment
            notes: Additional notes
            metadata: Additional metadata
            
        Returns:
            Created experiment
            
        Raises:
            ValueError: If the dataset is not found
        """
        # Check that the dataset exists
        dataset = await self.repository.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        # Generate experiment ID
        experiment_id = str(uuid.uuid4())
        
        # Create storage directory
        experiment_path = os.path.join(self.model_storage_path, "experiments", experiment_id)
        os.makedirs(experiment_path, exist_ok=True)
        
        # Initialize parameters if not provided
        if parameters is None:
            parameters = {}
            
        # Initialize tags if not provided
        if tags is None:
            tags = []
            
        # Initialize metadata if not provided
        if metadata is None:
            metadata = {}
            
        # Add storage location to metadata
        metadata["storage_path"] = experiment_path
        
        # Create the experiment object
        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            status=ExperimentStatus.CREATED,
            model_type=model_type,
            model_framework=model_framework,
            algorithm=algorithm,
            dataset_id=dataset_id,
            parameters=parameters,
            metrics={},
            artifacts={},
            tags=tags,
            created_at=datetime.utcnow(),
            created_by=created_by,
            notes=notes,
            metadata=metadata
        )
        
        # Save to repository
        await self.repository.create_experiment(experiment)
        
        logger.info(f"Created experiment {experiment_id}: {name}")
        
        return experiment
        
    async def get_experiment(self, experiment_id: str) -> Experiment:
        """
        Get an experiment by ID.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Experiment object
            
        Raises:
            ValueError: If the experiment is not found
        """
        experiment = await self.repository.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        return experiment
        
    async def list_experiments(
        self,
        filter_criteria: Optional[ExperimentFilter] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Experiment]:
        """
        List experiments with optional filtering.
        
        Args:
            filter_criteria: Criteria to filter experiments
            limit: Maximum number of experiments to return
            offset: Number of experiments to skip
            
        Returns:
            List of experiments matching the criteria
        """
        return await self.repository.list_experiments(
            filter_criteria=filter_criteria,
            limit=limit,
            offset=offset
        )
        
    async def update_experiment(
        self,
        experiment_id: str,
        **updates
    ) -> Experiment:
        """
        Update an experiment.
        
        Args:
            experiment_id: ID of the experiment to update
            **updates: Fields to update
            
        Returns:
            Updated experiment
            
        Raises:
            ValueError: If the experiment is not found
        """
        # Get the current experiment
        experiment = await self.get_experiment(experiment_id)
        
        # Update fields
        for key, value in updates.items():
            if hasattr(experiment, key):
                setattr(experiment, key, value)
                
        # Save to repository
        await self.repository.update_experiment(experiment)
        
        logger.info(f"Updated experiment {experiment_id}")
        
        return experiment
        
    # Training run methods
    
    async def create_training_run(
        self,
        experiment_id: str,
        parameters: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TrainingRun:
        """
        Create a new training run for an experiment.
        
        Args:
            experiment_id: ID of the parent experiment
            parameters: Training parameters (overrides experiment parameters)
            notes: Additional notes
            metadata: Additional metadata
            
        Returns:
            Created training run
            
        Raises:
            ValueError: If the experiment is not found
        """
        # Get the experiment
        experiment = await self.get_experiment(experiment_id)
        
        # Get existing training runs to determine run number
        existing_runs = await self.repository.list_training_runs(experiment_id)
        run_number = len(existing_runs) + 1
        
        # Generate run ID
        run_id = str(uuid.uuid4())
        
        # Create storage directory
        run_path = os.path.join(
            self.model_storage_path, "experiments", experiment_id, f"run_{run_number}"
        )
        os.makedirs(run_path, exist_ok=True)
        
        # Combine parameters from experiment and run
        combined_params = experiment.parameters.copy()
        if parameters:
            combined_params.update(parameters)
            
        # Initialize metadata if not provided
        if metadata is None:
            metadata = {}
            
        # Add storage location to metadata
        metadata["storage_path"] = run_path
        
        # Create the training run object
        training_run = TrainingRun(
            id=run_id,
            experiment_id=experiment_id,
            run_number=run_number,
            status=ExperimentStatus.CREATED,
            parameters=combined_params,
            metrics={},
            artifacts={},
            notes=notes,
            metadata=metadata
        )
        
        # Save to repository
        await self.repository.create_training_run(training_run)
        
        logger.info(f"Created training run {run_id} for experiment {experiment_id}")
        
        return training_run
        
    async def get_training_run(self, run_id: str) -> TrainingRun:
        """
        Get a training run by ID.
        
        Args:
            run_id: ID of the training run
            
        Returns:
            Training run object
            
        Raises:
            ValueError: If the training run is not found
        """
        training_run = await self.repository.get_training_run(run_id)
        if not training_run:
            raise ValueError(f"Training run {run_id} not found")
        return training_run
        
    async def list_training_runs(
        self,
        experiment_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[TrainingRun]:
        """
        List training runs for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            limit: Maximum number of runs to return
            offset: Number of runs to skip
            
        Returns:
            List of training runs
        """
        return await self.repository.list_training_runs(
            experiment_id=experiment_id,
            limit=limit,
            offset=offset
        )
        
    async def update_training_run(
        self,
        run_id: str,
        **updates
    ) -> TrainingRun:
        """
        Update a training run.
        
        Args:
            run_id: ID of the training run to update
            **updates: Fields to update
            
        Returns:
            Updated training run
            
        Raises:
            ValueError: If the training run is not found
        """
        # Get the current run
        training_run = await self.get_training_run(run_id)
        
        # Update fields
        for key, value in updates.items():
            if hasattr(training_run, key):
                setattr(training_run, key, value)
                
        # Save to repository
        await self.repository.update_training_run(training_run)
        
        logger.info(f"Updated training run {run_id}")
        
        return training_run
        
    async def add_metric_to_run(
        self,
        run_id: str,
        metric_name: str,
        metric_value: Union[float, Dict[str, float]],
        higher_is_better: bool = True
    ) -> MetricValue:
        """
        Add a metric to a training run.
        
        Args:
            run_id: ID of the training run
            metric_name: Name of the metric
            metric_value: Value of the metric
            higher_is_better: Whether higher values are better
            
        Returns:
            Created metric value
            
        Raises:
            ValueError: If the training run is not found
        """
        # Get the current run
        training_run = await self.get_training_run(run_id)
        
        # Create metric value
        metric = MetricValue(
            name=metric_name,
            value=metric_value,
            higher_is_better=higher_is_better,
            timestamp=datetime.utcnow()
        )
        
        # Add to metrics
        if training_run.metrics is None:
            training_run.metrics = {}
            
        training_run.metrics[metric_name] = metric
        
        # Save to repository
        await self.repository.update_training_run(training_run)
        
        logger.info(f"Added metric {metric_name}={metric_value} to run {run_id}")
        
        return metric
        
    # Model methods
    
    async def create_model(
        self,
        name: str,
        version: str,
        description: Optional[str],
        experiment_id: str,
        training_run_id: Optional[str],
        model_artifacts: Dict[str, str],
        features: List[str],
        target: Optional[str],
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Model:
        """
        Create a new model from a training run.
        
        Args:
            name: Model name
            version: Model version
            description: Model description
            experiment_id: ID of the experiment that produced this model
            training_run_id: ID of the specific training run (optional)
            model_artifacts: Dictionary mapping artifact names to file paths
            features: List of input feature names
            target: Target variable name (optional)
            tags: Tags for organization
            metadata: Additional metadata
            
        Returns:
            Created model
            
        Raises:
            ValueError: If the experiment or training run is not found
        """
        # Get the experiment
        experiment = await self.get_experiment(experiment_id)
        
        # Get the training run if provided
        if training_run_id:
            training_run = await self.get_training_run(training_run_id)
            parameters = training_run.parameters
            metrics = training_run.metrics
        else:
            # Use experiment parameters and metrics if no specific run
            parameters = experiment.parameters
            metrics = experiment.metrics
            
        # Generate model ID
        model_id = str(uuid.uuid4())
        
        # Create storage directory
        model_path = os.path.join(self.model_storage_path, "models", model_id)
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize tags if not provided
        if tags is None:
            tags = []
            
        # Initialize metadata if not provided
        if metadata is None:
            metadata = {}
            
        # Add storage location to metadata
        metadata["storage_path"] = model_path
        
        # Create the model object
        model = Model(
            id=model_id,
            name=name,
            version=version,
            description=description,
            experiment_id=experiment_id,
            training_run_id=training_run_id,
            model_type=experiment.model_type,
            model_framework=experiment.model_framework,
            algorithm=experiment.algorithm,
            parameters=parameters,
            metrics=metrics,
            artifacts=model_artifacts,
            features=features,
            target=target,
            created_at=datetime.utcnow(),
            is_published=False,
            tags=tags,
            metadata=metadata
        )
        
        # Save to repository
        await self.repository.create_model(model)
        
        logger.info(f"Created model {model_id}: {name} v{version}")
        
        return model
        
    async def get_model(self, model_id: str) -> Model:
        """
        Get a model by ID.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model object
            
        Raises:
            ValueError: If the model is not found
        """
        model = await self.repository.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        return model
        
    async def list_models(
        self,
        experiment_id: Optional[str] = None,
        is_published: Optional[bool] = None,
        name_filter: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Model]:
        """
        List models with optional filtering.
        
        Args:
            experiment_id: Filter by experiment ID
            is_published: Filter by publication status
            name_filter: Filter by name (partial match)
            limit: Maximum number of models to return
            offset: Number of models to skip
            
        Returns:
            List of models matching the criteria
        """
        return await self.repository.list_models(
            experiment_id=experiment_id,
            is_published=is_published,
            name_filter=name_filter,
            limit=limit,
            offset=offset
        )
        
    async def update_model(
        self,
        model_id: str,
        **updates
    ) -> Model:
        """
        Update a model.
        
        Args:
            model_id: ID of the model to update
            **updates: Fields to update
            
        Returns:
            Updated model
            
        Raises:
            ValueError: If the model is not found
        """
        # Get the current model
        model = await self.get_model(model_id)
        
        # Update fields
        for key, value in updates.items():
            if hasattr(model, key):
                setattr(model, key, value)
                
        # Update the timestamp
        model.updated_at = datetime.utcnow()
                
        # Save to repository
        await self.repository.update_model(model)
        
        logger.info(f"Updated model {model_id}")
        
        return model
        
    async def publish_model(self, model_id: str) -> Model:
        """
        Publish a model for production use.
        
        Args:
            model_id: ID of the model to publish
            
        Returns:
            Updated model
            
        Raises:
            ValueError: If the model is not found
        """
        # Get the current model
        model = await self.get_model(model_id)
        
        # Set published flag
        model.is_published = True
        model.updated_at = datetime.utcnow()
                
        # Save to repository
        await self.repository.update_model(model)
        
        logger.info(f"Published model {model_id}")
        
        return model
        
    async def unpublish_model(self, model_id: str) -> Model:
        """
        Unpublish a model.
        
        Args:
            model_id: ID of the model to unpublish
            
        Returns:
            Updated model
            
        Raises:
            ValueError: If the model is not found
        """
        # Get the current model
        model = await self.get_model(model_id)
        
        # Set published flag
        model.is_published = False
        model.updated_at = datetime.utcnow()
                
        # Save to repository
        await self.repository.update_model(model)
        
        logger.info(f"Unpublished model {model_id}")
        
        return model
        
    # Prediction methods
    
    async def record_prediction(
        self,
        model_id: str,
        input_data: Dict[str, Any],
        prediction_result: Any,
        confidence: Optional[float] = None,
        actual_value: Optional[Any] = None,
        additional_outputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Prediction:
        """
        Record a prediction made by a model.
        
        Args:
            model_id: ID of the model that made the prediction
            input_data: Input data for the prediction
            prediction_result: Prediction result
            confidence: Confidence score (0-1)
            actual_value: Actual value (if available)
            additional_outputs: Additional outputs
            metadata: Additional metadata
            
        Returns:
            Created prediction record
            
        Raises:
            ValueError: If the model is not found
        """
        # Check that the model exists
        model = await self.get_model(model_id)
        
        # Initialize additional outputs if not provided
        if additional_outputs is None:
            additional_outputs = {}
            
        # Initialize metadata if not provided
        if metadata is None:
            metadata = {}
            
        # Create prediction object
        prediction = Prediction(
            model_id=model_id,
            input_data=input_data,
            prediction=prediction_result,
            confidence=confidence,
            actual_value=actual_value,
            additional_outputs=additional_outputs,
            metadata=metadata
        )
        
        # Save to repository
        await self.repository.create_prediction(prediction)
        
        return prediction
        
    async def list_predictions(
        self,
        model_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Prediction]:
        """
        List predictions for a model.
        
        Args:
            model_id: ID of the model
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of predictions to return
            offset: Number of predictions to skip
            
        Returns:
            List of predictions
        """
        return await self.repository.list_predictions(
            model_id=model_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset
        )