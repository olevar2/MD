"""
Adapter for the Market Analysis Service.

This module provides an adapter for the Market Analysis Service, implementing
the IMarketAnalysisService interface.
"""
from typing import Dict, Any, List, Optional
import logging
import httpx
from datetime import datetime
from common_lib.interfaces.market_analysis_service_interface import IMarketAnalysisService
from common_lib.resilience.decorators import with_circuit_breaker, with_retry, with_timeout
from common_lib.resilience.factory import create_standard_resilience_config

logger = logging.getLogger(__name__)


class MarketAnalysisAdapter(IMarketAnalysisService):
    """Adapter for the Market Analysis Service."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Initialize the MarketAnalysisAdapter.

        Args:
            base_url: The base URL of the Market Analysis Service
            timeout: The timeout for HTTP requests in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.resilience_config = create_standard_resilience_config(
            service_name="market-analysis-service",
            timeout_seconds=timeout
        )
        logger.info(f"Initialized MarketAnalysisAdapter with base URL: {base_url}")

    @with_circuit_breaker("market-analysis-service")
    @with_retry("market-analysis-service")
    @with_timeout("market-analysis-service")
    async def analyze_market(self, 
                            symbol: str,
                            timeframe: str,
                            data: Dict[str, Any],
                            analysis_types: List[str],
                            config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive market analysis for the specified symbol and timeframe.

        Args:
            symbol: The symbol to analyze
            timeframe: The timeframe to analyze
            data: The market data to analyze
            analysis_types: List of analysis types to perform
            config: Optional configuration parameters

        Returns:
            A dictionary containing the analysis results
        """
        url = f"{self.base_url}/api/v1/analyze"
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data,
            "analysis_types": analysis_types,
            "config": config or {}
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("market-analysis-service")
    @with_retry("market-analysis-service")
    @with_timeout("market-analysis-service")
    async def detect_patterns(self, 
                             symbol: str,
                             timeframe: str,
                             data: Dict[str, Any],
                             pattern_types: Optional[List[str]] = None,
                             config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect chart patterns in the market data.

        Args:
            symbol: The symbol to analyze
            timeframe: The timeframe to analyze
            data: The market data to analyze
            pattern_types: Optional list of pattern types to detect
            config: Optional configuration parameters

        Returns:
            A dictionary containing the detected patterns
        """
        url = f"{self.base_url}/api/v1/patterns"
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data,
            "pattern_types": pattern_types or [],
            "config": config or {}
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("market-analysis-service")
    @with_retry("market-analysis-service")
    @with_timeout("market-analysis-service")
    async def detect_support_resistance(self, 
                                       symbol: str,
                                       timeframe: str,
                                       data: Dict[str, Any],
                                       methods: Optional[List[str]] = None,
                                       config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect support and resistance levels in the market data.

        Args:
            symbol: The symbol to analyze
            timeframe: The timeframe to analyze
            data: The market data to analyze
            methods: Optional list of detection methods to use
            config: Optional configuration parameters

        Returns:
            A dictionary containing the support and resistance levels
        """
        url = f"{self.base_url}/api/v1/support-resistance"
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data,
            "methods": methods or [],
            "config": config or {}
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("market-analysis-service")
    @with_retry("market-analysis-service")
    @with_timeout("market-analysis-service")
    async def detect_market_regime(self, 
                                  symbol: str,
                                  timeframe: str,
                                  data: Dict[str, Any],
                                  config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect the current market regime based on the market data.

        Args:
            symbol: The symbol to analyze
            timeframe: The timeframe to analyze
            data: The market data to analyze
            config: Optional configuration parameters

        Returns:
            A dictionary containing the market regime information
        """
        url = f"{self.base_url}/api/v1/market-regime"
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data,
            "config": config or {}
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("market-analysis-service")
    @with_retry("market-analysis-service")
    @with_timeout("market-analysis-service")
    async def get_regime_history(self,
                                symbol: str,
                                timeframe: str,
                                limit: Optional[int] = 10,
                                config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get historical regime data for a specific symbol and timeframe.

        Args:
            symbol: The symbol to analyze
            timeframe: The timeframe to analyze
            limit: Maximum number of historical regimes to return
            config: Optional configuration parameters

        Returns:
            A list of dictionaries containing historical regime information
        """
        url = f"{self.base_url}/api/v1/market-regime/history"
        params = {
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("market-analysis-service")
    @with_retry("market-analysis-service")
    @with_timeout("market-analysis-service")
    async def analyze_correlation(self, 
                                 symbols: List[str],
                                 timeframe: str,
                                 data: Dict[str, Dict[str, Any]],
                                 method: Optional[str] = "pearson",
                                 config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze correlation between multiple symbols.

        Args:
            symbols: List of symbols to analyze
            timeframe: The timeframe to analyze
            data: Dictionary of market data for each symbol
            method: Correlation method to use
            config: Optional configuration parameters

        Returns:
            A dictionary containing the correlation analysis
        """
        url = f"{self.base_url}/api/v1/correlation"
        payload = {
            "symbols": symbols,
            "timeframe": timeframe,
            "data": data,
            "method": method,
            "config": config or {}
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("market-analysis-service")
    @with_retry("market-analysis-service")
    @with_timeout("market-analysis-service")
    async def find_optimal_market_conditions(self,
                                           tool_id: str,
                                           min_sample_size: int = 10,
                                           timeframe: Optional[str] = None,
                                           instrument: Optional[str] = None,
                                           from_date: Optional[datetime] = None,
                                           to_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Find the optimal market conditions for a specific tool.

        Args:
            tool_id: The ID of the tool to analyze
            min_sample_size: Minimum sample size for analysis
            timeframe: Optional timeframe filter
            instrument: Optional instrument filter
            from_date: Optional start date filter
            to_date: Optional end date filter

        Returns:
            A dictionary containing the optimal market conditions
        """
        url = f"{self.base_url}/api/v1/optimal-conditions"
        payload = {
            "tool_id": tool_id,
            "min_sample_size": min_sample_size
        }
        
        if timeframe:
            payload["timeframe"] = timeframe
        if instrument:
            payload["instrument"] = instrument
        if from_date:
            payload["from_date"] = from_date.isoformat()
        if to_date:
            payload["to_date"] = to_date.isoformat()
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("market-analysis-service")
    @with_retry("market-analysis-service")
    @with_timeout("market-analysis-service")
    async def recommend_tools_for_current_regime(self,
                                               current_regime: str,
                                               instrument: Optional[str] = None,
                                               timeframe: Optional[str] = None,
                                               min_sample_size: int = 10,
                                               min_win_rate: float = 50.0,
                                               top_n: int = 3) -> Dict[str, Any]:
        """
        Recommend the best trading tools for the current market regime.

        Args:
            current_regime: The current market regime
            instrument: Optional instrument filter
            timeframe: Optional timeframe filter
            min_sample_size: Minimum sample size for analysis
            min_win_rate: Minimum win rate for recommended tools
            top_n: Number of top tools to recommend

        Returns:
            A dictionary containing the recommended tools
        """
        url = f"{self.base_url}/api/v1/recommend-tools"
        payload = {
            "current_regime": current_regime,
            "min_sample_size": min_sample_size,
            "min_win_rate": min_win_rate,
            "top_n": top_n
        }
        
        if timeframe:
            payload["timeframe"] = timeframe
        if instrument:
            payload["instrument"] = instrument
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()