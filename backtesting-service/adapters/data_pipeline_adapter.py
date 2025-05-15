from typing import Dict, List, Any, Optional
import os
import httpx
from common_lib.resilience.decorators import with_standard_resilience

class DataPipelineAdapter:
    """
    Adapter for interacting with the data pipeline service.
    """
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the adapter with the data pipeline service URL.
        """
        self.base_url = base_url or os.environ.get("DATA_PIPELINE_URL", "http://data-pipeline-service:8000")
        self.timeout = int(os.environ.get("DATA_PIPELINE_TIMEOUT", "60"))
    
    @with_standard_resilience()
    async def get_historical_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Get historical data from the data pipeline service.
        """
        params = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/historical-data", params=params)
            response.raise_for_status()
            return response.json()
    
    @with_standard_resilience()
    async def get_available_symbols(self) -> List[str]:
        """
        Get available symbols from the data pipeline service.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/symbols")
            response.raise_for_status()
            return response.json()
    
    @with_standard_resilience()
    async def get_available_timeframes(self) -> List[str]:
        """
        Get available timeframes from the data pipeline service.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/timeframes")
            response.raise_for_status()
            return response.json()
    
    @with_standard_resilience()
    async def get_data_quality_metrics(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Get data quality metrics from the data pipeline service.
        """
        params = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/data-quality", params=params)
            response.raise_for_status()
            return response.json()