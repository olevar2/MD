"""
ML Workbench Interfaces Module

This module provides interfaces for ML workbench functionality used across services,
helping to break circular dependencies between services.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field


class ModelOptimizationType(str, Enum):
    """Types of model optimization"""
    HYPERPARAMETER = "hyperparameter"
    ARCHITECTURE = "architecture"
    FEATURE_SELECTION = "feature_selection"
    ENSEMBLE = "ensemble"
    REINFORCEMENT = "reinforcement"
    CUSTOM = "custom"


@dataclass
class OptimizationConfig:
    """Configuration for model optimization"""
    model_id: str
    optimization_type: ModelOptimizationType
    parameters: Dict[str, Any]
    max_iterations: int
    target_metric: str
    constraints: Optional[Dict[str, Any]] = None
    timeout_minutes: Optional[int] = None


@dataclass
class OptimizationResult:
    """Result of model optimization"""
    model_id: str
    optimization_id: str
    best_parameters: Dict[str, Any]
    best_score: float
    iterations_completed: int
    timestamp: datetime
    history: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class IModelOptimizationService(ABC):
    """Interface for model optimization services"""
    
    @abstractmethod
    async def optimize_model(
        self,
        config: OptimizationConfig
    ) -> str:
        """
        Start a model optimization job.
        
        Args:
            config: Optimization configuration
            
        Returns:
            Optimization job ID
        """
        pass
    
    @abstractmethod
    async def get_optimization_status(
        self,
        optimization_id: str
    ) -> Dict[str, Any]:
        """
        Get the status of an optimization job.
        
        Args:
            optimization_id: Optimization job ID
            
        Returns:
            Dictionary with optimization status
        """
        pass
    
    @abstractmethod
    async def get_optimization_result(
        self,
        optimization_id: str
    ) -> OptimizationResult:
        """
        Get the result of an optimization job.
        
        Args:
            optimization_id: Optimization job ID
            
        Returns:
            Optimization result
        """
        pass
    
    @abstractmethod
    async def cancel_optimization(
        self,
        optimization_id: str
    ) -> bool:
        """
        Cancel an optimization job.
        
        Args:
            optimization_id: Optimization job ID
            
        Returns:
            Success flag
        """
        pass


class IModelRegistryService(ABC):
    """Interface for model registry services"""
    
    @abstractmethod
    async def register_model(
        self,
        model_id: str,
        model_type: str,
        version: str,
        metadata: Dict[str, Any],
        artifacts_path: str
    ) -> Dict[str, Any]:
        """
        Register a model in the registry.
        
        Args:
            model_id: Model identifier
            model_type: Type of model
            version: Model version
            metadata: Model metadata
            artifacts_path: Path to model artifacts
            
        Returns:
            Dictionary with registration details
        """
        pass
    
    @abstractmethod
    async def get_model_info(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_id: Model identifier
            version: Optional model version
            
        Returns:
            Dictionary with model information
        """
        pass
    
    @abstractmethod
    async def list_models(
        self,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List models in the registry.
        
        Args:
            model_type: Optional filter by model type
            tags: Optional filter by tags
            limit: Maximum number of results
            offset: Result offset for pagination
            
        Returns:
            List of model information dictionaries
        """
        pass
    
    @abstractmethod
    async def delete_model(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id: Model identifier
            version: Optional model version (if None, deletes all versions)
            
        Returns:
            Success flag
        """
        pass


class IReinforcementLearningService(ABC):
    """Interface for reinforcement learning services"""
    
    @abstractmethod
    async def train_rl_model(
        self,
        model_id: str,
        environment_config: Dict[str, Any],
        agent_config: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> str:
        """
        Train a reinforcement learning model.
        
        Args:
            model_id: Model identifier
            environment_config: Environment configuration
            agent_config: Agent configuration
            training_config: Training configuration
            
        Returns:
            Training job ID
        """
        pass
    
    @abstractmethod
    async def get_training_status(
        self,
        training_id: str
    ) -> Dict[str, Any]:
        """
        Get the status of a training job.
        
        Args:
            training_id: Training job ID
            
        Returns:
            Dictionary with training status
        """
        pass
    
    @abstractmethod
    async def get_rl_model_performance(
        self,
        model_id: str,
        environment_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get performance metrics for a reinforcement learning model.
        
        Args:
            model_id: Model identifier
            environment_config: Environment configuration
            
        Returns:
            Dictionary with performance metrics
        """
        pass
    
    @abstractmethod
    async def get_rl_model_action(
        self,
        model_id: str,
        state: Dict[str, Any],
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get an action from a reinforcement learning model.
        
        Args:
            model_id: Model identifier
            state: Current environment state
            version: Optional model version
            
        Returns:
            Dictionary with action and metadata
        """
        pass
