"""
Experiment Repository Module.

Handles the storage and retrieval of experiments, models, datasets, and predictions.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from ml_workbench_service.models.experiment_models import (
    Dataset, Experiment, ExperimentFilter, ExperimentStatus, 
    Model, Prediction, TrainingRun
)


class ExperimentRepository:
    """Repository for managing ML experiments, models, and related entities."""
    
    def __init__(self, db_client):
        """Initialize the repository with a database client.
        
        Args:
            db_client: A database client instance for database operations
        """
        self.db = db_client
        
    async def create_dataset(self, dataset: Dataset) -> Dataset:
        """Create a new dataset record.
        
        Args:
            dataset: The dataset to create
            
        Returns:
            The created dataset with ID
        """
        if not dataset.id:
            dataset.id = str(uuid.uuid4())
        
        # Store in database
        await self.db.datasets.insert_one(dataset.dict())
        return dataset
    
    async def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """Retrieve a dataset by ID.
        
        Args:
            dataset_id: ID of the dataset to retrieve
            
        Returns:
            The dataset if found, None otherwise
        """
        result = await self.db.datasets.find_one({"id": dataset_id})
        if result:
            return Dataset(**result)
        return None
    
    async def list_datasets(self, limit: int = 100, offset: int = 0) -> List[Dataset]:
        """List datasets with pagination.
        
        Args:
            limit: Maximum number of datasets to return
            offset: Number of datasets to skip
            
        Returns:
            List of datasets
        """
        cursor = self.db.datasets.find({}).sort("created_at", -1).skip(offset).limit(limit)
        return [Dataset(**doc) async for doc in cursor]
    
    async def create_experiment(self, experiment: Experiment) -> Experiment:
        """Create a new experiment.
        
        Args:
            experiment: The experiment to create
            
        Returns:
            The created experiment with ID
        """
        if not experiment.id:
            experiment.id = str(uuid.uuid4())
        
        experiment.created_at = datetime.utcnow()
        experiment.updated_at = experiment.created_at
        
        # Store in database
        await self.db.experiments.insert_one(experiment.dict())
        return experiment
    
    async def update_experiment(self, experiment: Experiment) -> Experiment:
        """Update an existing experiment.
        
        Args:
            experiment: The experiment with updated fields
            
        Returns:
            The updated experiment
        """
        experiment.updated_at = datetime.utcnow()
        
        # Update in database
        await self.db.experiments.replace_one(
            {"id": experiment.id},
            experiment.dict()
        )
        return experiment
    
    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Retrieve an experiment by ID.
        
        Args:
            experiment_id: ID of the experiment to retrieve
            
        Returns:
            The experiment if found, None otherwise
        """
        result = await self.db.experiments.find_one({"id": experiment_id})
        if result:
            return Experiment(**result)
        return None
    
    async def list_experiments(
        self, 
        filter_criteria: Optional[ExperimentFilter] = None,
        limit: int = 100, 
        offset: int = 0
    ) -> List[Experiment]:
        """List experiments with filtering and pagination.
        
        Args:
            filter_criteria: Optional criteria to filter experiments
            limit: Maximum number of experiments to return
            offset: Number of experiments to skip
            
        Returns:
            List of filtered experiments
        """
        query = {}
        
        if filter_criteria:
            if filter_criteria.name:
                query["name"] = {"$regex": filter_criteria.name, "$options": "i"}
            
            if filter_criteria.status:
                query["status"] = {"$in": [s.value for s in filter_criteria.status]}
                
            if filter_criteria.model_type:
                query["model_type"] = {"$in": [t.value for t in filter_criteria.model_type]}
                
            if filter_criteria.model_framework:
                query["model_framework"] = {"$in": [f.value for f in filter_criteria.model_framework]}
                
            if filter_criteria.algorithm:
                query["algorithm"] = {"$in": [a.value for a in filter_criteria.algorithm]}
                
            if filter_criteria.dataset_id:
                query["dataset_id"] = filter_criteria.dataset_id
                
            if filter_criteria.tags:
                query["tags"] = {"$all": filter_criteria.tags}
                
            if filter_criteria.created_after or filter_criteria.created_before:
                date_query = {}
                if filter_criteria.created_after:
                    date_query["$gte"] = filter_criteria.created_after
                if filter_criteria.created_before:
                    date_query["$lte"] = filter_criteria.created_before
                query["created_at"] = date_query
                
            if filter_criteria.created_by:
                query["created_by"] = filter_criteria.created_by
        
        cursor = self.db.experiments.find(query).sort("created_at", -1).skip(offset).limit(limit)
        return [Experiment(**doc) async for doc in cursor]
    
    async def create_training_run(self, run: TrainingRun) -> TrainingRun:
        """Create a new training run.
        
        Args:
            run: The training run to create
            
        Returns:
            The created training run with ID
        """
        if not run.id:
            run.id = str(uuid.uuid4())
        
        # Store in database
        await self.db.training_runs.insert_one(run.dict())
        return run
    
    async def update_training_run(self, run: TrainingRun) -> TrainingRun:
        """Update an existing training run.
        
        Args:
            run: The training run with updated fields
            
        Returns:
            The updated training run
        """
        # Update in database
        await self.db.training_runs.replace_one(
            {"id": run.id},
            run.dict()
        )
        return run
    
    async def get_training_run(self, run_id: str) -> Optional[TrainingRun]:
        """Retrieve a training run by ID.
        
        Args:
            run_id: ID of the training run to retrieve
            
        Returns:
            The training run if found, None otherwise
        """
        result = await self.db.training_runs.find_one({"id": run_id})
        if result:
            return TrainingRun(**result)
        return None
    
    async def list_training_runs(
        self, 
        experiment_id: str,
        limit: int = 100, 
        offset: int = 0
    ) -> List[TrainingRun]:
        """List training runs for an experiment with pagination.
        
        Args:
            experiment_id: ID of the experiment
            limit: Maximum number of runs to return
            offset: Number of runs to skip
            
        Returns:
            List of training runs
        """
        cursor = self.db.training_runs.find(
            {"experiment_id": experiment_id}
        ).sort("run_number", -1).skip(offset).limit(limit)
        
        return [TrainingRun(**doc) async for doc in cursor]
    
    async def create_model(self, model: Model) -> Model:
        """Create a new model.
        
        Args:
            model: The model to create
            
        Returns:
            The created model with ID
        """
        if not model.id:
            model.id = str(uuid.uuid4())
        
        model.created_at = datetime.utcnow()
        model.updated_at = model.created_at
        
        # Store in database
        await self.db.models.insert_one(model.dict())
        return model
    
    async def update_model(self, model: Model) -> Model:
        """Update an existing model.
        
        Args:
            model: The model with updated fields
            
        Returns:
            The updated model
        """
        model.updated_at = datetime.utcnow()
        
        # Update in database
        await self.db.models.replace_one(
            {"id": model.id},
            model.dict()
        )
        return model
    
    async def get_model(self, model_id: str) -> Optional[Model]:
        """Retrieve a model by ID.
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            The model if found, None otherwise
        """
        result = await self.db.models.find_one({"id": model_id})
        if result:
            return Model(**result)
        return None
    
    async def list_models(
        self, 
        experiment_id: Optional[str] = None,
        is_published: Optional[bool] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100, 
        offset: int = 0
    ) -> List[Model]:
        """List models with filtering and pagination.
        
        Args:
            experiment_id: Optional experiment ID filter
            is_published: Optional publication status filter
            tags: Optional list of tags to match
            limit: Maximum number of models to return
            offset: Number of models to skip
            
        Returns:
            List of filtered models
        """
        query = {}
        
        if experiment_id:
            query["experiment_id"] = experiment_id
            
        if is_published is not None:
            query["is_published"] = is_published
            
        if tags:
            query["tags"] = {"$all": tags}
            
        cursor = self.db.models.find(query).sort("created_at", -1).skip(offset).limit(limit)
        return [Model(**doc) async for doc in cursor]
    
    async def create_prediction(self, prediction: Prediction) -> Prediction:
        """Create a new prediction record.
        
        Args:
            prediction: The prediction to create
            
        Returns:
            The created prediction with ID
        """
        if not prediction.id:
            prediction.id = str(uuid.uuid4())
        
        # Store in database
        await self.db.predictions.insert_one(prediction.dict())
        return prediction
    
    async def get_prediction(self, prediction_id: str) -> Optional[Prediction]:
        """Retrieve a prediction by ID.
        
        Args:
            prediction_id: ID of the prediction to retrieve
            
        Returns:
            The prediction if found, None otherwise
        """
        result = await self.db.predictions.find_one({"id": prediction_id})
        if result:
            return Prediction(**result)
        return None
    
    async def list_predictions(
        self, 
        model_id: str,
        limit: int = 100, 
        offset: int = 0
    ) -> List[Prediction]:
        """List predictions for a model with pagination.
        
        Args:
            model_id: ID of the model
            limit: Maximum number of predictions to return
            offset: Number of predictions to skip
            
        Returns:
            List of predictions
        """
        cursor = self.db.predictions.find(
            {"model_id": model_id}
        ).sort("timestamp", -1).skip(offset).limit(limit)
        
        return [Prediction(**doc) async for doc in cursor]