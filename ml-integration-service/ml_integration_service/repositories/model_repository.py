"""
Model Repository for the ML Integration Service.

This module provides functionality for accessing and managing model data
in the model repository.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import pandas as pd
import numpy as np
import aiohttp
import asyncio
from urllib.parse import urljoin

from ml_integration_service.config.enhanced_settings import enhanced_settings

logger = logging.getLogger(__name__)


class ModelRepository:
    """Repository for accessing and managing model data."""
    
    def __init__(self):
        """Initialize the model repository."""
        self.ml_workbench_url = enhanced_settings.ML_WORKBENCH_API_URL
        
    async def get_model_training_data(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get training data for a model.
        
        Args:
            model_id: ID of the model
            version: Version of the model
            
        Returns:
            DataFrame with training data
        """
        async with aiohttp.ClientSession() as session:
            url = urljoin(self.ml_workbench_url, f"/api/v1/models/{model_id}/training-data")
            params = {}
            if version:
                params["version"] = version
                
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                return pd.DataFrame(data["training_data"])
                
    async def get_training_dataset(self, dataset_id: str) -> pd.DataFrame:
        """
        Get a training dataset by ID.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            DataFrame with dataset data
        """
        async with aiohttp.ClientSession() as session:
            url = urljoin(self.ml_workbench_url, f"/api/v1/datasets/{dataset_id}")
            
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                
                return pd.DataFrame(data["dataset_data"])
                
    async def get_model_inference_data(
        self,
        model_id: str,
        version: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get inference data for a model.
        
        Args:
            model_id: ID of the model
            version: Version of the model
            start_time: Start time for the data
            end_time: End time for the data
            
        Returns:
            DataFrame with inference data
        """
        async with aiohttp.ClientSession() as session:
            url = urljoin(self.ml_workbench_url, f"/api/v1/models/{model_id}/inference-data")
            params = {}
            if version:
                params["version"] = version
            if start_time:
                params["start_time"] = start_time.isoformat()
            if end_time:
                params["end_time"] = end_time.isoformat()
                
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                return pd.DataFrame(data["inference_data"])
                
    async def update_model_training_data(
        self,
        model_id: str,
        field: str,
        value: Any,
        version: Optional[str] = None
    ) -> bool:
        """
        Update training data for a model.
        
        Args:
            model_id: ID of the model
            field: Field to update
            value: New value for the field
            version: Version of the model
            
        Returns:
            Whether the update was successful
        """
        async with aiohttp.ClientSession() as session:
            url = urljoin(self.ml_workbench_url, f"/api/v1/models/{model_id}/training-data")
            params = {}
            if version:
                params["version"] = version
                
            data = {
                "field": field,
                "value": value
            }
                
            async with session.patch(url, params=params, json=data) as response:
                response.raise_for_status()
                result = await response.json()
                
                return result["success"]
                
    async def update_model_inference_data(
        self,
        model_id: str,
        field: str,
        value: Any,
        version: Optional[str] = None
    ) -> bool:
        """
        Update inference data for a model.
        
        Args:
            model_id: ID of the model
            field: Field to update
            value: New value for the field
            version: Version of the model
            
        Returns:
            Whether the update was successful
        """
        async with aiohttp.ClientSession() as session:
            url = urljoin(self.ml_workbench_url, f"/api/v1/models/{model_id}/inference-data")
            params = {}
            if version:
                params["version"] = version
                
            data = {
                "field": field,
                "value": value
            }
                
            async with session.patch(url, params=params, json=data) as response:
                response.raise_for_status()
                result = await response.json()
                
                return result["success"]
