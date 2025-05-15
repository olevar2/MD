"""
ML Workbench Service Interface.

This module defines the interfaces for interacting with the ML Workbench Service.
These interfaces allow other services to use ML Workbench functionality without
direct dependencies, breaking circular dependencies.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import pandas as pd


class IExperimentManager(ABC):
    """Interface for experiment management."""
    
    @abstractmethod
    async def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new experiment.
        
        Args:
            name: Name of the experiment
            description: Optional description
            tags: Optional tags
            metadata: Optional metadata
            
        Returns:
            Dictionary with experiment details
        """
        pass
    
    @abstractmethod
    async def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get details of an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Dictionary with experiment details
        """
        pass
    
    @abstractmethod
    async def list_experiments(
        self,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List experiments.
        
        Args:
            tags: Optional tags to filter by
            status: Optional status to filter by
            limit: Maximum number of results
            offset: Result offset
            
        Returns:
            Dictionary with experiments and pagination information
        """
        pass
    
    @abstractmethod
    async def update_experiment(
        self,
        experiment_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update an experiment.
        
        Args:
            experiment_id: ID of the experiment
            updates: Dictionary of updates to apply
            
        Returns:
            Dictionary with updated experiment details
        """
        pass
    
    @abstractmethod
    async def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            True if the experiment was deleted successfully
        """
        pass


class IModelEvaluator(ABC):
    """Interface for model evaluation."""
    
    @abstractmethod
    async def evaluate_model(
        self,
        model_id: str,
        dataset_id: str,
        metrics: List[str],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a model on a dataset.
        
        Args:
            model_id: ID of the model
            dataset_id: ID of the dataset
            metrics: List of metrics to calculate
            parameters: Optional evaluation parameters
            
        Returns:
            Dictionary with evaluation results
        """
        pass
    
    @abstractmethod
    async def compare_models(
        self,
        model_ids: List[str],
        dataset_id: str,
        metrics: List[str],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models on a dataset.
        
        Args:
            model_ids: IDs of the models
            dataset_id: ID of the dataset
            metrics: List of metrics to calculate
            parameters: Optional comparison parameters
            
        Returns:
            Dictionary with comparison results
        """
        pass
    
    @abstractmethod
    async def get_evaluation_history(
        self,
        model_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get evaluation history for a model.
        
        Args:
            model_id: ID of the model
            limit: Maximum number of results
            
        Returns:
            List of evaluation results
        """
        pass


class IDatasetManager(ABC):
    """Interface for dataset management."""
    
    @abstractmethod
    async def create_dataset(
        self,
        name: str,
        data: Union[pd.DataFrame, Dict[str, Any]],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new dataset.
        
        Args:
            name: Name of the dataset
            data: Dataset data
            description: Optional description
            tags: Optional tags
            metadata: Optional metadata
            
        Returns:
            Dictionary with dataset details
        """
        pass
    
    @abstractmethod
    async def get_dataset(
        self,
        dataset_id: str,
        include_data: bool = False
    ) -> Dict[str, Any]:
        """
        Get details of a dataset.
        
        Args:
            dataset_id: ID of the dataset
            include_data: Whether to include the dataset data
            
        Returns:
            Dictionary with dataset details
        """
        pass
    
    @abstractmethod
    async def list_datasets(
        self,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List datasets.
        
        Args:
            tags: Optional tags to filter by
            limit: Maximum number of results
            offset: Result offset
            
        Returns:
            Dictionary with datasets and pagination information
        """
        pass
    
    @abstractmethod
    async def update_dataset(
        self,
        dataset_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update a dataset.
        
        Args:
            dataset_id: ID of the dataset
            updates: Dictionary of updates to apply
            
        Returns:
            Dictionary with updated dataset details
        """
        pass
    
    @abstractmethod
    async def delete_dataset(self, dataset_id: str) -> bool:
        """
        Delete a dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            True if the dataset was deleted successfully
        """
        pass
