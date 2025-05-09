"""
ML Workbench Adapter Module

This module provides adapter implementations for ML workbench interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
import asyncio
import json
import os
import httpx

from common_lib.ml.workbench_interfaces import (
    IModelOptimizationService,
    IModelRegistryService,
    IReinforcementLearningService,
    ModelOptimizationType,
    OptimizationConfig,
    OptimizationResult
)
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class ModelOptimizationServiceAdapter(IModelOptimizationService):
    """
    Adapter for model optimization service that implements the common interface.
    
    This adapter can either use a direct API connection to the ML workbench service
    or provide standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Get ML workbench service URL from config or environment
        ml_workbench_base_url = self.config.get(
            "ml_workbench_base_url", 
            os.environ.get("ML_WORKBENCH_BASE_URL", "http://ml-workbench-service:8000")
        )
        
        # Set up the client with resolved URL
        self.client = httpx.AsyncClient(
            base_url=f"{ml_workbench_base_url.rstrip('/')}/api/v1",
            timeout=30.0
        )
    
    async def optimize_model(
        self,
        config: OptimizationConfig
    ) -> str:
        """Start a model optimization job."""
        try:
            # Prepare request data
            request_data = {
                "model_id": config.model_id,
                "optimization_type": config.optimization_type.value,
                "parameters": config.parameters,
                "max_iterations": config.max_iterations,
                "target_metric": config.target_metric
            }
            
            if config.constraints:
                request_data["constraints"] = config.constraints
                
            if config.timeout_minutes:
                request_data["timeout_minutes"] = config.timeout_minutes
            
            # Send request
            response = await self.client.post(
                "/optimization/start",
                json=request_data
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            return result.get("optimization_id", "")
            
        except Exception as e:
            logger.error(f"Error starting model optimization: {str(e)}")
            
            # Return empty string as fallback
            return ""
    
    async def get_optimization_status(
        self,
        optimization_id: str
    ) -> Dict[str, Any]:
        """Get the status of an optimization job."""
        try:
            # Send request
            response = await self.client.get(
                f"/optimization/{optimization_id}/status"
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting optimization status: {str(e)}")
            
            # Return fallback status
            return {
                "optimization_id": optimization_id,
                "status": "error",
                "error": str(e),
                "progress": 0.0,
                "is_fallback": True
            }
    
    async def get_optimization_result(
        self,
        optimization_id: str
    ) -> OptimizationResult:
        """Get the result of an optimization job."""
        try:
            # Send request
            response = await self.client.get(
                f"/optimization/{optimization_id}/result"
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Convert to OptimizationResult
            return OptimizationResult(
                model_id=result.get("model_id", ""),
                optimization_id=optimization_id,
                best_parameters=result.get("best_parameters", {}),
                best_score=result.get("best_score", 0.0),
                iterations_completed=result.get("iterations_completed", 0),
                timestamp=datetime.fromisoformat(result.get("timestamp")) if "timestamp" in result else datetime.now(),
                history=result.get("history"),
                metadata=result.get("metadata")
            )
            
        except Exception as e:
            logger.error(f"Error getting optimization result: {str(e)}")
            
            # Return fallback result
            return OptimizationResult(
                model_id="",
                optimization_id=optimization_id,
                best_parameters={},
                best_score=0.0,
                iterations_completed=0,
                timestamp=datetime.now(),
                history=None,
                metadata={"error": str(e), "is_fallback": True}
            )
    
    async def cancel_optimization(
        self,
        optimization_id: str
    ) -> bool:
        """Cancel an optimization job."""
        try:
            # Send request
            response = await self.client.post(
                f"/optimization/{optimization_id}/cancel"
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"Error canceling optimization: {str(e)}")
            
            # Return failure as fallback
            return False


class ModelRegistryServiceAdapter(IModelRegistryService):
    """
    Adapter for model registry service that implements the common interface.
    
    This adapter can either use a direct API connection to the ML workbench service
    or provide standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Get ML workbench service URL from config or environment
        ml_workbench_base_url = self.config.get(
            "ml_workbench_base_url", 
            os.environ.get("ML_WORKBENCH_BASE_URL", "http://ml-workbench-service:8000")
        )
        
        # Set up the client with resolved URL
        self.client = httpx.AsyncClient(
            base_url=f"{ml_workbench_base_url.rstrip('/')}/api/v1",
            timeout=30.0
        )
        
        # Cache for model info
        self.model_info_cache = {}
        self.cache_ttl = self.config.get("cache_ttl_minutes", 15)  # Cache TTL in minutes
    
    async def register_model(
        self,
        model_id: str,
        model_type: str,
        version: str,
        metadata: Dict[str, Any],
        artifacts_path: str
    ) -> Dict[str, Any]:
        """Register a model in the registry."""
        try:
            # Prepare request data
            request_data = {
                "model_id": model_id,
                "model_type": model_type,
                "version": version,
                "metadata": metadata,
                "artifacts_path": artifacts_path
            }
            
            # Send request
            response = await self.client.post(
                "/models/register",
                json=request_data
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            
            # Return fallback registration details
            return {
                "model_id": model_id,
                "version": version,
                "status": "error",
                "error": str(e),
                "is_fallback": True
            }
    
    async def get_model_info(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get information about a model."""
        try:
            # Check cache first
            cache_key = f"{model_id}_{version or 'latest'}"
            if cache_key in self.model_info_cache:
                cache_entry = self.model_info_cache[cache_key]
                cache_age = (datetime.now() - cache_entry["timestamp"]).total_seconds() / 60
                if cache_age < self.cache_ttl:
                    return cache_entry["info"]
            
            # Prepare query parameters
            params = {}
            if version:
                params["version"] = version
            
            # Send request
            response = await self.client.get(
                f"/models/{model_id}",
                params=params
            )
            response.raise_for_status()
            
            # Parse response
            info = response.json()
            
            # Update cache
            self.model_info_cache[cache_key] = {
                "info": info,
                "timestamp": datetime.now()
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            
            # Return fallback model info
            return {
                "model_id": model_id,
                "version": version or "latest",
                "status": "error",
                "error": str(e),
                "is_fallback": True
            }
    
    async def list_models(
        self,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List models in the registry."""
        try:
            # Prepare query parameters
            params = {
                "limit": limit,
                "offset": offset
            }
            if model_type:
                params["model_type"] = model_type
            if tags:
                params["tags"] = ",".join(tags)
            
            # Send request
            response = await self.client.get(
                "/models",
                params=params
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            
            # Return empty list as fallback
            return []
    
    async def delete_model(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> bool:
        """Delete a model from the registry."""
        try:
            # Prepare query parameters
            params = {}
            if version:
                params["version"] = version
            
            # Send request
            response = await self.client.delete(
                f"/models/{model_id}",
                params=params
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Clear cache entries for this model
            cache_keys_to_remove = [k for k in self.model_info_cache.keys() if k.startswith(f"{model_id}_")]
            for key in cache_keys_to_remove:
                self.model_info_cache.pop(key, None)
            
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"Error deleting model: {str(e)}")
            
            # Return failure as fallback
            return False


class ReinforcementLearningServiceAdapter(IReinforcementLearningService):
    """
    Adapter for reinforcement learning service that implements the common interface.
    
    This adapter can either use a direct API connection to the ML workbench service
    or provide standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Get ML workbench service URL from config or environment
        ml_workbench_base_url = self.config.get(
            "ml_workbench_base_url", 
            os.environ.get("ML_WORKBENCH_BASE_URL", "http://ml-workbench-service:8000")
        )
        
        # Set up the client with resolved URL
        self.client = httpx.AsyncClient(
            base_url=f"{ml_workbench_base_url.rstrip('/')}/api/v1",
            timeout=30.0
        )
    
    async def train_rl_model(
        self,
        model_id: str,
        environment_config: Dict[str, Any],
        agent_config: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> str:
        """Train a reinforcement learning model."""
        try:
            # Prepare request data
            request_data = {
                "model_id": model_id,
                "environment_config": environment_config,
                "agent_config": agent_config,
                "training_config": training_config
            }
            
            # Send request
            response = await self.client.post(
                "/rl/train",
                json=request_data
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            return result.get("training_id", "")
            
        except Exception as e:
            logger.error(f"Error training RL model: {str(e)}")
            
            # Return empty string as fallback
            return ""
    
    async def get_training_status(
        self,
        training_id: str
    ) -> Dict[str, Any]:
        """Get the status of a training job."""
        try:
            # Send request
            response = await self.client.get(
                f"/rl/training/{training_id}/status"
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting training status: {str(e)}")
            
            # Return fallback status
            return {
                "training_id": training_id,
                "status": "error",
                "error": str(e),
                "progress": 0.0,
                "is_fallback": True
            }
    
    async def get_rl_model_performance(
        self,
        model_id: str,
        environment_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get performance metrics for a reinforcement learning model."""
        try:
            # Prepare request data
            request_data = {
                "environment_config": environment_config
            }
            
            # Send request
            response = await self.client.post(
                f"/rl/models/{model_id}/performance",
                json=request_data
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting RL model performance: {str(e)}")
            
            # Return fallback performance metrics
            return {
                "model_id": model_id,
                "metrics": {},
                "error": str(e),
                "is_fallback": True
            }
    
    async def get_rl_model_action(
        self,
        model_id: str,
        state: Dict[str, Any],
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get an action from a reinforcement learning model."""
        try:
            # Prepare request data
            request_data = {
                "state": state
            }
            
            # Prepare query parameters
            params = {}
            if version:
                params["version"] = version
            
            # Send request
            response = await self.client.post(
                f"/rl/models/{model_id}/action",
                params=params,
                json=request_data
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting RL model action: {str(e)}")
            
            # Return fallback action
            return {
                "model_id": model_id,
                "action": None,
                "confidence": 0.0,
                "error": str(e),
                "is_fallback": True
            }
