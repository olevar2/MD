"""
Analysis Engine Adapter Module

This module provides adapter implementations for analysis engine interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
import asyncio
import json
import os
import httpx

from common_lib.analysis.interfaces import (
    IMarketRegimeAnalyzer,
    IPatternRecognitionService,
    ITechnicalAnalysisService,
    MarketRegimeType,
    PatternType,
    IndicatorType
)
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class MarketRegimeAnalyzerAdapter(IMarketRegimeAnalyzer):
    """
    Adapter for market regime analyzer that implements the common interface.
    
    This adapter can either use a direct API connection to the analysis engine service
    or provide standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
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
    
    async def detect_market_regime(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int = 100,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Detect the current market regime."""
        try:
            # Prepare query parameters
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "lookback_bars": lookback_bars
            }
            if methods:
                params["methods"] = ",".join(methods)
            
            # Send request
            response = await self.client.get(
                "/market-regime/detect",
                params=params
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            
            # Return fallback regime
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "regime": MarketRegimeType.UNKNOWN.value,
                "confidence": 0.0,
                "sub_regimes": {},
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "is_fallback": True
            }
    
    async def get_regime_history(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        methods: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get historical market regime data."""
        try:
            # Prepare query parameters
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_time": start_time.isoformat()
            }
            if end_time:
                params["end_time"] = end_time.isoformat()
            if methods:
                params["methods"] = ",".join(methods)
            
            # Send request
            response = await self.client.get(
                "/market-regime/history",
                params=params
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting regime history: {str(e)}")
            
            # Return empty list as fallback
            return []
    
    async def get_regime_transition_probabilities(
        self,
        symbol: str,
        timeframe: str,
        lookback_days: int = 30
    ) -> Dict[str, Dict[str, float]]:
        """Get transition probabilities between market regimes."""
        try:
            # Prepare query parameters
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "lookback_days": lookback_days
            }
            
            # Send request
            response = await self.client.get(
                "/market-regime/transitions",
                params=params
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting regime transition probabilities: {str(e)}")
            
            # Return empty dict as fallback
            return {}


class PatternRecognitionServiceAdapter(IPatternRecognitionService):
    """
    Adapter for pattern recognition service that implements the common interface.
    
    This adapter can either use a direct API connection to the analysis engine service
    or provide standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
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
    
    async def detect_patterns(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int = 100,
        pattern_types: Optional[List[PatternType]] = None
    ) -> List[Dict[str, Any]]:
        """Detect chart patterns."""
        try:
            # Prepare query parameters
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "lookback_bars": lookback_bars
            }
            if pattern_types:
                params["pattern_types"] = ",".join([pt.value for pt in pattern_types])
            
            # Send request
            response = await self.client.get(
                "/patterns/detect",
                params=params
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
            
            # Return empty list as fallback
            return []
    
    async def get_pattern_statistics(
        self,
        symbol: str,
        timeframe: str,
        pattern_type: PatternType,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """Get statistics for a specific pattern type."""
        try:
            # Prepare query parameters
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "pattern_type": pattern_type.value,
                "lookback_days": lookback_days
            }
            
            # Send request
            response = await self.client.get(
                "/patterns/statistics",
                params=params
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting pattern statistics: {str(e)}")
            
            # Return fallback statistics
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "pattern_type": pattern_type.value,
                "occurrences": 0,
                "success_rate": 0.0,
                "average_return": 0.0,
                "error": str(e),
                "is_fallback": True
            }


class TechnicalAnalysisServiceAdapter(ITechnicalAnalysisService):
    """
    Adapter for technical analysis service that implements the common interface.
    
    This adapter can either use a direct API connection to the analysis engine service
    or provide standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
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
    
    async def calculate_indicators(
        self,
        symbol: str,
        timeframe: str,
        indicators: List[Dict[str, Any]],
        lookback_bars: int = 100
    ) -> Dict[str, List[float]]:
        """Calculate technical indicators."""
        try:
            # Prepare request data
            request_data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "indicators": indicators,
                "lookback_bars": lookback_bars
            }
            
            # Send request
            response = await self.client.post(
                "/indicators/calculate",
                json=request_data
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            
            # Return empty dict as fallback
            return {}
    
    async def get_indicator_signals(
        self,
        symbol: str,
        timeframe: str,
        indicators: List[Dict[str, Any]],
        lookback_bars: int = 100
    ) -> List[Dict[str, Any]]:
        """Get trading signals from technical indicators."""
        try:
            # Prepare request data
            request_data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "indicators": indicators,
                "lookback_bars": lookback_bars
            }
            
            # Send request
            response = await self.client.post(
                "/indicators/signals",
                json=request_data
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting indicator signals: {str(e)}")
            
            # Return empty list as fallback
            return []
    
    async def get_indicator_performance(
        self,
        symbol: str,
        timeframe: str,
        indicator_type: IndicatorType,
        parameters: Dict[str, Any],
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """Get performance metrics for a specific indicator."""
        try:
            # Prepare request data
            request_data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "indicator_type": indicator_type.value,
                "parameters": parameters,
                "lookback_days": lookback_days
            }
            
            # Send request
            response = await self.client.post(
                "/indicators/performance",
                json=request_data
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting indicator performance: {str(e)}")
            
            # Return fallback performance metrics
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "indicator_type": indicator_type.value,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_return": 0.0,
                "error": str(e),
                "is_fallback": True
            }
