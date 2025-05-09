"""
Analysis Engine Adapter Module

This module provides adapters for analysis engine functionality,
helping to break circular dependencies between services.
"""
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import os
import httpx
import asyncio
import json

from common_lib.ml.interfaces import IAnalysisEngineClient

logger = logging.getLogger(__name__)


class AnalysisEngineClientAdapter(IAnalysisEngineClient):
    """
    Adapter for analysis engine client functionality.
    
    This adapter can either use a direct connection to the analysis engine service
    or provide standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Get analysis engine service URL from config or environment
        analysis_engine_base_url = self.config.get(
            "analysis_engine_base_url", 
            os.environ.get("ANALYSIS_ENGINE_BASE_URL", "http://analysis-engine-service:8000")
        )
        
        # Set up the client with resolved URL
        self.client = httpx.AsyncClient(
            base_url=f"{analysis_engine_base_url.rstrip('/')}/api/v1",
            timeout=30.0
        )
        
        # Cache for market data
        self.market_data_cache = {}
        self.cache_ttl = self.config.get("cache_ttl_minutes", 15)  # Cache TTL in minutes
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        bars: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get market data for a symbol."""
        try:
            # Check cache first
            cache_key = f"market_data_{symbol}_{timeframe}_{bars}"
            if cache_key in self.market_data_cache:
                cache_entry = self.market_data_cache[cache_key]
                if datetime.now() < cache_entry["expiry"]:
                    return cache_entry["data"]
            
            # Prepare request parameters
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "bars": bars
            }
            
            # Add optional parameters if provided
            if start_time:
                params["start_time"] = start_time.isoformat()
            if end_time:
                params["end_time"] = end_time.isoformat()
            
            # Make API request
            response = await self.client.get("/market-data", params=params)
            response.raise_for_status()
            
            # Parse response
            market_data = response.json()
            
            # Cache the result
            self.market_data_cache[cache_key] = {
                "data": market_data,
                "expiry": datetime.now() + timedelta(minutes=self.cache_ttl)
            }
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            
            # Return fallback market data
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "data": []
            }
    
    async def get_technical_indicators(
        self,
        symbol: str,
        timeframe: str,
        indicators: List[Dict[str, Any]],
        bars: int = 100
    ) -> Dict[str, Any]:
        """Get technical indicators for a symbol."""
        try:
            # Prepare request data
            request_data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "indicators": indicators,
                "bars": bars
            }
            
            # Make API request
            response = await self.client.post("/technical-indicators", json=request_data)
            response.raise_for_status()
            
            # Parse response
            indicator_data = response.json()
            
            return indicator_data
            
        except Exception as e:
            self.logger.error(f"Error getting technical indicators: {str(e)}")
            
            # Return fallback indicators
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "indicators": {}
            }
    
    async def get_market_regime(
        self,
        symbol: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Get market regime for a symbol."""
        try:
            # Prepare request parameters
            params = {
                "symbol": symbol,
                "timeframe": timeframe
            }
            
            # Make API request
            response = await self.client.get("/market-regime", params=params)
            response.raise_for_status()
            
            # Parse response
            regime_data = response.json()
            
            return regime_data
            
        except Exception as e:
            self.logger.error(f"Error getting market regime: {str(e)}")
            
            # Return fallback regime
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "regime": "unknown",
                "confidence": 0.5,
                "volatility": "medium"
            }
    
    async def get_support_resistance_levels(
        self,
        symbol: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Get support and resistance levels for a symbol."""
        try:
            # Prepare request parameters
            params = {
                "symbol": symbol,
                "timeframe": timeframe
            }
            
            # Make API request
            response = await self.client.get("/support-resistance", params=params)
            response.raise_for_status()
            
            # Parse response
            levels_data = response.json()
            
            return levels_data
            
        except Exception as e:
            self.logger.error(f"Error getting support resistance levels: {str(e)}")
            
            # Return fallback levels
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "support": [],
                "resistance": []
            }
