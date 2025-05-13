"""
Data Service Integration Module

This module handles data retrieval and integration with various data sources
required by the ML Integration Service.
"""
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import aiohttp
import asyncio
from urllib.parse import urljoin

from ml_integration_service.config.enhanced_settings import enhanced_settings
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)

class DataService:
    """Service for retrieving and managing data from various sources."""
    
    def __init__(self):
    """
      init  .
    
    """

        self.ml_workbench_url = enhanced_settings.ML_WORKBENCH_API_URL
        self.data_pipeline_url = enhanced_settings.DATA_PIPELINE_API_URL
        
    async def get_model_performance_data(
        self,
        model_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Retrieve model performance data for visualization."""
        async with aiohttp.ClientSession() as session:
            url = urljoin(self.ml_workbench_url, f"/api/v1/models/{model_id}/performance")
            params = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            }
            
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                return pd.DataFrame(data["performance_data"])
                
    async def get_feature_importance(self, model_id: str) -> Dict[str, float]:
        """Retrieve feature importance data for a model."""
        async with aiohttp.ClientSession() as session:
            url = urljoin(self.ml_workbench_url, f"/api/v1/models/{model_id}/feature-importance")
            
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                
                return data["feature_importance"]
                
    async def get_regime_performance(
        self,
        model_id: str,
        regime_info: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Retrieve regime-specific performance data."""
        async with aiohttp.ClientSession() as session:
            url = urljoin(self.ml_workbench_url, f"/api/v1/models/{model_id}/regime-performance")
            
            async with session.get(url, json=regime_info) as response:
                response.raise_for_status()
                data = await response.json()
                
                return pd.DataFrame(data["regime_performance"])
                
    async def get_historical_parameters(
        self,
        strategy_id: str,
        lookback_days: int = 90
    ) -> pd.DataFrame:
        """Retrieve historical parameter data for optimization."""
        async with aiohttp.ClientSession() as session:
            url = urljoin(self.data_pipeline_url, f"/api/v1/strategies/{strategy_id}/parameters")
            params = {"lookback_days": lookback_days}
            
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                return pd.DataFrame(data["parameter_history"])
                
    async def get_test_market_data(
        self, 
        window_params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Retrieve market data for stress testing."""
        async with aiohttp.ClientSession() as session:
            url = urljoin(self.data_pipeline_url, "/api/v1/market-data")
            
            async with session.get(url, json=window_params) as response:
                response.raise_for_status()
                data = await response.json()
                
                return pd.DataFrame(data["market_data"])
                
    async def get_model_predictions(
        self,
        model_id: str,
        input_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Get model predictions for testing."""
        async with aiohttp.ClientSession() as session:
            url = urljoin(self.ml_workbench_url, f"/api/v1/models/{model_id}/predict")
            
            # Convert DataFrame to dict for JSON serialization
            data = input_data.to_dict(orient='records')
            
            async with session.post(url, json={"data": data}) as response:
                response.raise_for_status()
                result = await response.json()
                
                return pd.DataFrame(result["predictions"])
