"""
Feature Service for the ML Integration Service.

This service provides functionality for accessing and managing features
required by the ML Integration Service.
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
from common_lib.resilience.circuit_breaker import (
    create_circuit_breaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen
)

logger = logging.getLogger(__name__)


class FeatureService:
    """Service for accessing and managing features."""

    def __init__(self):
        """Initialize the feature service."""
        self.feature_store_url = enhanced_settings.FEATURE_STORE_API_URL

        # Create circuit breakers for different operations
        self.training_data_circuit = create_circuit_breaker(
            service_name="ml_integration_service",
            resource_name="feature_store_training_data",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30.0,
                timeout=10.0
            )
        )

        self.inference_data_circuit = create_circuit_breaker(
            service_name="ml_integration_service",
            resource_name="feature_store_inference_data",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30.0,
                timeout=10.0
            )
        )

    async def get_cached_training_data(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get cached training data for a model.

        Args:
            model_id: ID of the model
            version: Version of the model

        Returns:
            DataFrame with training data

        Raises:
            CircuitBreakerOpen: If the circuit breaker is open
            aiohttp.ClientError: If there is an error communicating with the feature store
        """
        # Define the function to execute with circuit breaker protection
        async def fetch_training_data():
            async with aiohttp.ClientSession() as session:
                url = urljoin(self.feature_store_url, f"/api/v1/cache/training-data")
                params = {
                    "model_id": model_id
                }
                if version:
                    params["version"] = version

                try:
                    async with session.get(url, params=params) as response:
                        response.raise_for_status()
                        data = await response.json()

                        return pd.DataFrame(data["training_data"])
                except aiohttp.ClientError as e:
                    logger.error(f"Error fetching training data from feature store: {str(e)}")
                    raise

        try:
            # Execute the function with circuit breaker protection
            return await self.training_data_circuit.execute(fetch_training_data)
        except CircuitBreakerOpen as e:
            logger.error(f"Circuit breaker open for feature store training data: {str(e)}")
            # Return empty DataFrame as fallback
            return pd.DataFrame()

    async def get_cached_inference_data(
        self,
        model_id: str,
        version: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get cached inference data for a model.

        Args:
            model_id: ID of the model
            version: Version of the model
            start_time: Start time for the data
            end_time: End time for the data

        Returns:
            DataFrame with inference data

        Raises:
            CircuitBreakerOpen: If the circuit breaker is open
            aiohttp.ClientError: If there is an error communicating with the feature store
        """
        # Define the function to execute with circuit breaker protection
        async def fetch_inference_data():
            async with aiohttp.ClientSession() as session:
                url = urljoin(self.feature_store_url, f"/api/v1/cache/inference-data")
                params = {
                    "model_id": model_id
                }
                if version:
                    params["version"] = version
                if start_time:
                    params["start_time"] = start_time.isoformat()
                if end_time:
                    params["end_time"] = end_time.isoformat()

                try:
                    async with session.get(url, params=params) as response:
                        response.raise_for_status()
                        data = await response.json()

                        return pd.DataFrame(data["inference_data"])
                except aiohttp.ClientError as e:
                    logger.error(f"Error fetching inference data from feature store: {str(e)}")
                    raise

        try:
            # Execute the function with circuit breaker protection
            return await self.inference_data_circuit.execute(fetch_inference_data)
        except CircuitBreakerOpen as e:
            logger.error(f"Circuit breaker open for feature store inference data: {str(e)}")
            # Return empty DataFrame as fallback
            return pd.DataFrame()

    async def update_cached_training_data(
        self,
        model_id: str,
        field: str,
        value: Any,
        version: Optional[str] = None
    ) -> bool:
        """
        Update cached training data for a model.

        Args:
            model_id: ID of the model
            field: Field to update
            value: New value for the field
            version: Version of the model

        Returns:
            Whether the update was successful

        Raises:
            CircuitBreakerOpen: If the circuit breaker is open
            aiohttp.ClientError: If there is an error communicating with the feature store
        """
        # Define the function to execute with circuit breaker protection
        async def update_training_data():
            async with aiohttp.ClientSession() as session:
                url = urljoin(self.feature_store_url, f"/api/v1/cache/training-data")
                params = {
                    "model_id": model_id
                }
                if version:
                    params["version"] = version

                data = {
                    "field": field,
                    "value": value
                }

                try:
                    async with session.patch(url, params=params, json=data) as response:
                        response.raise_for_status()
                        result = await response.json()

                        return result["success"]
                except aiohttp.ClientError as e:
                    logger.error(f"Error updating training data in feature store: {str(e)}")
                    raise

        try:
            # Execute the function with circuit breaker protection
            return await self.training_data_circuit.execute(update_training_data)
        except CircuitBreakerOpen as e:
            logger.error(f"Circuit breaker open for feature store training data update: {str(e)}")
            # Return False as fallback
            return False

    async def update_cached_inference_data(
        self,
        model_id: str,
        field: str,
        value: Any,
        version: Optional[str] = None
    ) -> bool:
        """
        Update cached inference data for a model.

        Args:
            model_id: ID of the model
            field: Field to update
            value: New value for the field
            version: Version of the model

        Returns:
            Whether the update was successful

        Raises:
            CircuitBreakerOpen: If the circuit breaker is open
            aiohttp.ClientError: If there is an error communicating with the feature store
        """
        # Define the function to execute with circuit breaker protection
        async def update_inference_data():
            async with aiohttp.ClientSession() as session:
                url = urljoin(self.feature_store_url, f"/api/v1/cache/inference-data")
                params = {
                    "model_id": model_id
                }
                if version:
                    params["version"] = version

                data = {
                    "field": field,
                    "value": value
                }

                try:
                    async with session.patch(url, params=params, json=data) as response:
                        response.raise_for_status()
                        result = await response.json()

                        return result["success"]
                except aiohttp.ClientError as e:
                    logger.error(f"Error updating inference data in feature store: {str(e)}")
                    raise

        try:
            # Execute the function with circuit breaker protection
            return await self.inference_data_circuit.execute(update_inference_data)
        except CircuitBreakerOpen as e:
            logger.error(f"Circuit breaker open for feature store inference data update: {str(e)}")
            # Return False as fallback
            return False
