import aiohttp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from analysis_coordinator_service.models.coordinator_models import (
    AnalysisTaskStatusEnum,
    AnalysisTaskResult
)
from analysis_coordinator_service.utils.resilience import with_retry, with_circuit_breaker

logger = logging.getLogger(__name__)

class MarketAnalysisAdapter:
    """
    Adapter for communicating with the market analysis service.
    """
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    async def _make_request(self, method: str, url: str, payload: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make an HTTP request to the market analysis service.
        """
        async with aiohttp.ClientSession() as session:
            if method == "POST":
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Request failed with status {response.status}: {error_text}")
                        raise Exception(f"Market analysis request failed: {error_text}")
                    
                    return await response.json()
            elif method == "GET":
                async with session.get(url) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Request failed with status {response.status}: {error_text}")
                        raise Exception(f"Market analysis request failed: {error_text}")
                    
                    return await response.json()
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
        
    @with_retry(max_retries=3, backoff_factor=0.5)
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def analyze_market(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Perform market analysis.
        """
        if parameters is None:
            parameters = {}
            
        url = f"{self.base_url}/api/v1/market-analysis/analyze"
        
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "parameters": parameters
        }
        
        if end_date:
            payload["end_date"] = end_date.isoformat()
            
        logger.info(f"Sending market analysis request to {url}")
        
        return await self._make_request("POST", url, payload)
                
    @with_retry(max_retries=3, backoff_factor=0.5)
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def get_patterns(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        patterns: List[str] = None
    ) -> Dict[str, Any]:
        """
        Recognize patterns in market data.
        """
        if patterns is None:
            patterns = []
            
        url = f"{self.base_url}/api/v1/market-analysis/patterns"
        
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "patterns": patterns
        }
        
        if end_date:
            payload["end_date"] = end_date.isoformat()
            
        logger.info(f"Sending pattern recognition request to {url}")
        
        return await self._make_request("POST", url, payload)
                
    @with_retry(max_retries=3, backoff_factor=0.5)
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def get_support_resistance(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Identify support and resistance levels.
        """
        if parameters is None:
            parameters = {}
            
        url = f"{self.base_url}/api/v1/market-analysis/support-resistance"
        
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "parameters": parameters
        }
        
        if end_date:
            payload["end_date"] = end_date.isoformat()
            
        logger.info(f"Sending support/resistance request to {url}")
        
        return await self._make_request("POST", url, payload)
                
    @with_retry(max_retries=3, backoff_factor=0.5)
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def get_market_regime(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Detect market regime.
        """
        if parameters is None:
            parameters = {}
            
        url = f"{self.base_url}/api/v1/market-analysis/market-regime"
        
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "parameters": parameters
        }
        
        if end_date:
            payload["end_date"] = end_date.isoformat()
            
        logger.info(f"Sending market regime request to {url}")
        
        return await self._make_request("POST", url, payload)
                
    @with_retry(max_retries=3, backoff_factor=0.5)
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def get_correlations(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        symbols: List[str] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze correlations between symbols.
        """
        if symbols is None:
            symbols = []
            
        if parameters is None:
            parameters = {}
            
        url = f"{self.base_url}/api/v1/market-analysis/correlation"
        
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "symbols": symbols,
            "parameters": parameters
        }
        
        if end_date:
            payload["end_date"] = end_date.isoformat()
            
        logger.info(f"Sending correlation analysis request to {url}")
        
        return await self._make_request("POST", url, payload)
