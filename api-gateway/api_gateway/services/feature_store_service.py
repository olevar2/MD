"""
Feature Store Service Client

This module provides a client for the Feature Store Service.
"""

import logging
import json
from typing import Dict, Any, Optional, List

import httpx
from httpx import AsyncClient, Response

from common_lib.config.config_manager import ConfigManager
from common_lib.resilience.circuit_breaker import CircuitBreaker
from common_lib.resilience.retry import retry


class FeatureStoreService:
    """
    Client for the Feature Store Service.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the client.
        
        Args:
            logger: Logger to use (if None, creates a new logger)
        """
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config_manager = ConfigManager()
        
        # Get service client configuration
        try:
            service_clients = self.config_manager.get_service_clients_config()
            self.client_config = service_clients.feature_store_service
        except Exception as e:
            self.logger.warning(f"Error getting service client configuration: {str(e)}")
            self.client_config = None
        
        # Set default values
        self.base_url = getattr(self.client_config, "base_url", "http://localhost:8004") if self.client_config else "http://localhost:8004"
        self.timeout = getattr(self.client_config, "timeout", 30.0) if self.client_config else 30.0
        
        # Create circuit breaker
        self.circuit_breaker = CircuitBreaker(
            name="feature_store_service",
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exceptions=[httpx.HTTPError, httpx.TimeoutException],
            logger=self.logger
        )
    
    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Response:
        """
        Send a request to the service.
        
        Args:
            method: HTTP method
            path: Request path
            params: Query parameters
            json_data: JSON data
            headers: HTTP headers
            
        Returns:
            Response
            
        Raises:
            httpx.HTTPError: If the request fails
        """
        url = f"{self.base_url}{path}"
        
        # Add default headers
        headers = headers or {}
        headers["Content-Type"] = "application/json"
        
        # Send request with retry and circuit breaker
        @retry(
            retries=3,
            delay=1.0,
            backoff=2.0,
            exceptions=[httpx.HTTPError, httpx.TimeoutException]
        )
        async def send_request():
            async with AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    headers=headers
                )
                response.raise_for_status()
                return response
        
        return await self.circuit_breaker.execute(send_request)
    
    async def get_features(self) -> List[Dict[str, Any]]:
        """
        Get features.
        
        Returns:
            List of features
        """
        response = await self._request(
            "GET",
            "/api/v1/features"
        )
        return response.json()
    
    async def get_feature(self, feature: str) -> Dict[str, Any]:
        """
        Get feature.
        
        Args:
            feature: Feature name
            
        Returns:
            Feature
        """
        response = await self._request(
            "GET",
            f"/api/v1/features/{feature}"
        )
        return response.json()
    
    async def get_feature_data(
        self,
        feature: str,
        symbol: str,
        timeframe: str,
        parameters: Dict[str, Any],
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get feature data.
        
        Args:
            feature: Feature name
            symbol: Symbol
            timeframe: Timeframe
            parameters: Feature parameters
            start: Start timestamp in milliseconds
            end: End timestamp in milliseconds
            limit: Limit
            
        Returns:
            Feature data
        """
        json_data = {
            "feature": feature,
            "symbol": symbol,
            "timeframe": timeframe,
            "parameters": parameters
        }
        
        if start is not None:
            json_data["start"] = start
        if end is not None:
            json_data["end"] = end
        if limit is not None:
            json_data["limit"] = limit
        
        response = await self._request(
            "POST",
            "/api/v1/features/data",
            json_data=json_data
        )
        return response.json()
    
    async def get_feature_sets(self) -> List[Dict[str, Any]]:
        """
        Get feature sets.
        
        Returns:
            List of feature sets
        """
        response = await self._request(
            "GET",
            "/api/v1/feature-sets"
        )
        return response.json()
    
    async def get_feature_set(self, name: str) -> Dict[str, Any]:
        """
        Get feature set.
        
        Args:
            name: Feature set name
            
        Returns:
            Feature set
        """
        response = await self._request(
            "GET",
            f"/api/v1/feature-sets/{name}"
        )
        return response.json()
    
    async def create_feature_set(
        self,
        name: str,
        description: str,
        features: List[str]
    ) -> Dict[str, Any]:
        """
        Create feature set.
        
        Args:
            name: Feature set name
            description: Feature set description
            features: Features
            
        Returns:
            Feature set
        """
        json_data = {
            "name": name,
            "description": description,
            "features": features
        }
        
        response = await self._request(
            "POST",
            "/api/v1/feature-sets",
            json_data=json_data
        )
        return response.json()
    
    async def update_feature_set(
        self,
        name: str,
        description: str,
        features: List[str]
    ) -> Dict[str, Any]:
        """
        Update feature set.
        
        Args:
            name: Feature set name
            description: Feature set description
            features: Features
            
        Returns:
            Feature set
        """
        json_data = {
            "name": name,
            "description": description,
            "features": features
        }
        
        response = await self._request(
            "PUT",
            f"/api/v1/feature-sets/{name}",
            json_data=json_data
        )
        return response.json()
    
    async def delete_feature_set(self, name: str) -> Dict[str, Any]:
        """
        Delete feature set.
        
        Args:
            name: Feature set name
            
        Returns:
            Feature set
        """
        response = await self._request(
            "DELETE",
            f"/api/v1/feature-sets/{name}"
        )
        return response.json()