"""
Analysis Engine adapter implementation
"""
from typing import Dict, Any, List, Optional
import httpx
import pandas as pd
from datetime import datetime

from common_lib.analysis.interfaces import IAnalysisProvider
from common_lib.events.event_bus import EventBus, EventType
from common_lib.resilience.retry import async_retry

class AnalysisEngineAdapter(IAnalysisProvider):
    """Adapter for the Analysis Engine Service"""

    def __init__(self, base_url: str, api_key: str):
        """Initialize the adapter

        Args:
            base_url: Base URL of the analysis engine service
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.event_bus = EventBus()
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"X-Api-Key": self.api_key},
            timeout=60.0
        )

    @async_retry(max_retries=3)
    async def get_market_analysis(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        components: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get market analysis from the analysis engine service"""
        try:
            response = await self.client.post(
                "/api/v1/analysis/market",
                json={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "data": data.to_dict(orient="records"),
                    "components": components
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            await self.event_bus.publish(
                EventType.ANALYSIS_ERROR,
                {
                    "error": str(e),
                    "symbol": symbol,
                    "timeframe": timeframe
                }
            )
            raise

    @async_retry(max_retries=3)
    async def get_causal_analysis(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        variables: List[str]
    ) -> Dict[str, Any]:
        """Get causal analysis from the analysis engine service"""
        try:
            response = await self.client.post(
                "/api/v1/analysis/causal",
                json={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "data": data.to_dict(orient="records"),
                    "variables": variables
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            await self.event_bus.publish(
                EventType.ANALYSIS_ERROR,
                {
                    "error": str(e),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "analysis_type": "causal"
                }
            )
            raise

    @async_retry(max_retries=3)
    async def get_regime_analysis(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Get market regime analysis from the analysis engine service"""
        try:
            response = await self.client.post(
                "/api/v1/analysis/regime",
                json={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "data": data.to_dict(orient="records")
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            await self.event_bus.publish(
                EventType.ANALYSIS_ERROR,
                {
                    "error": str(e),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "analysis_type": "regime"
                }
            )
            raise
