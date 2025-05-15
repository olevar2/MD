from typing import Dict, List, Any, Optional
import os
import httpx
from common_lib.resilience.decorators import with_standard_resilience

class FeatureStoreAdapter:
    """
    Adapter for interacting with the feature store service.
    """
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the adapter with the feature store service URL.
        """
        self.base_url = base_url or os.environ.get("FEATURE_STORE_URL", "http://feature-store-service:8000")
        self.timeout = int(os.environ.get("FEATURE_STORE_TIMEOUT", "30"))
    
    @with_standard_resilience()
    async def get_feature_data(
        self, 
        feature_names: List[str], 
        start_time: Optional[str] = None, 
        end_time: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """
        Get feature data from the feature store.
        """
        params = {
            "feature_names": ",".join(feature_names)
        }
        
        if start_time:
            params["start_time"] = start_time
        
        if end_time:
            params["end_time"] = end_time
        
        if symbols:
            params["symbols"] = ",".join(symbols)
        
        if timeframes:
            params["timeframes"] = ",".join(timeframes)
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/features/data", params=params)
            response.raise_for_status()
            return response.json()
    
    @with_standard_resilience()
    async def get_feature_metadata(self, feature_names: List[str]) -> Dict[str, Any]:
        """
        Get metadata for the specified features.
        """
        params = {
            "feature_names": ",".join(feature_names)
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/features/metadata", params=params)
            response.raise_for_status()
            return response.json()
    
    @with_standard_resilience()
    async def get_feature_correlations(
        self, 
        feature_names: List[str],
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Get correlation matrix for the specified features.
        """
        params = {
            "feature_names": ",".join(feature_names)
        }
        
        if symbols:
            params["symbols"] = ",".join(symbols)
        
        if timeframes:
            params["timeframes"] = ",".join(timeframes)
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/features/correlations", params=params)
            response.raise_for_status()
            return response.json()