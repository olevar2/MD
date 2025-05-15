"""
Feature Store Adapter for Market Analysis Service.

This module provides an adapter for communicating with the Feature Store Service
to retrieve features for analysis.
"""
import logging
import uuid
from typing import Dict, List, Any, Optional
import httpx
import pandas as pd
from common_lib.resilience.decorators import (
    retry_with_backoff,
    circuit_breaker,
    timeout
)

logger = logging.getLogger(__name__)

class FeatureStoreAdapter:
    """
    Adapter for communicating with the Feature Store Service.
    """
    
    def __init__(self, base_url: str = "http://feature-store-service:8000"):
        """
        Initialize the Feature Store Adapter.
        
        Args:
            base_url: Base URL of the Feature Store Service
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        
    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @timeout(seconds=10)
    async def get_features(
        self,
        symbol: str,
        timeframe: str,
        feature_names: List[str],
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get features from the Feature Store Service.
        
        Args:
            symbol: Symbol to get features for
            timeframe: Timeframe to get features for
            feature_names: Names of features to get
            start_date: Start date for features
            end_date: End date for features
            
        Returns:
            DataFrame with features
        """
        try:
            request_id = str(uuid.uuid4())
            headers = {"X-Request-ID": request_id}
            
            payload = {
                "symbol": symbol,
                "timeframe": timeframe,
                "feature_names": feature_names,
                "start_date": start_date
            }
            
            if end_date:
                payload["end_date"] = end_date
                
            url = f"{self.base_url}/api/v1/features/batch"
            
            logger.info(f"Getting features for {symbol} {timeframe}")
            response = await self.client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data["features"])
            
            # Convert timestamp to datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                
            return df
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error when getting features: {e}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error when getting features: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error when getting features: {e}")
            raise
            
    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @timeout(seconds=10)
    async def get_available_features(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get available features from the Feature Store Service.
        
        Args:
            symbol: Optional symbol to filter features
            timeframe: Optional timeframe to filter features
            
        Returns:
            List of available features
        """
        try:
            request_id = str(uuid.uuid4())
            headers = {"X-Request-ID": request_id}
            
            params = {}
            if symbol:
                params["symbol"] = symbol
            if timeframe:
                params["timeframe"] = timeframe
                
            url = f"{self.base_url}/api/v1/features/available"
            
            logger.info("Getting available features")
            response = await self.client.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            return data["features"]
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error when getting available features: {e}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error when getting available features: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error when getting available features: {e}")
            raise
            
    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @timeout(seconds=10)
    async def get_feature_metadata(
        self,
        feature_name: str
    ) -> Dict[str, Any]:
        """
        Get metadata for a feature from the Feature Store Service.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Feature metadata
        """
        try:
            request_id = str(uuid.uuid4())
            headers = {"X-Request-ID": request_id}
            
            url = f"{self.base_url}/api/v1/features/metadata/{feature_name}"
            
            logger.info(f"Getting metadata for feature {feature_name}")
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error when getting feature metadata: {e}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error when getting feature metadata: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error when getting feature metadata: {e}")
            raise