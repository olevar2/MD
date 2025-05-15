"""
Adapter for the Causal Analysis Service.

This module provides an adapter for the Causal Analysis Service, implementing
the ICausalAnalysisService interface.
"""
from typing import Dict, Any, List, Optional
import logging
import httpx
from common_lib.interfaces.causal_analysis_service_interface import ICausalAnalysisService
from common_lib.resilience.decorators import with_circuit_breaker, with_retry, with_timeout
from common_lib.resilience.factory import create_standard_resilience_config

logger = logging.getLogger(__name__)


class CausalAnalysisAdapter(ICausalAnalysisService):
    """Adapter for the Causal Analysis Service."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Initialize the CausalAnalysisAdapter.

        Args:
            base_url: The base URL of the Causal Analysis Service
            timeout: The timeout for HTTP requests in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.resilience_config = create_standard_resilience_config(
            service_name="causal-analysis-service",
            timeout_seconds=timeout
        )
        logger.info(f"Initialized CausalAnalysisAdapter with base URL: {base_url}")

    @with_circuit_breaker("causal-analysis-service")
    @with_retry("causal-analysis-service")
    @with_timeout("causal-analysis-service")
    async def generate_causal_graph(self, 
                                   data: Dict[str, Any], 
                                   config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a causal graph from the provided data.

        Args:
            data: The data to analyze
            config: Optional configuration parameters

        Returns:
            A dictionary containing the causal graph and related information
        """
        url = f"{self.base_url}/api/v1/causal-graph"
        payload = {
            "data": data,
            "config": config or {}
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("causal-analysis-service")
    @with_retry("causal-analysis-service")
    @with_timeout("causal-analysis-service")
    async def analyze_intervention_effect(self, 
                                         data: Dict[str, Any], 
                                         intervention: Dict[str, Any],
                                         config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze the effect of an intervention on the system.

        Args:
            data: The data to analyze
            intervention: The intervention to apply
            config: Optional configuration parameters

        Returns:
            A dictionary containing the intervention effect analysis
        """
        url = f"{self.base_url}/api/v1/intervention-effect"
        payload = {
            "data": data,
            "intervention": intervention,
            "config": config or {}
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("causal-analysis-service")
    @with_retry("causal-analysis-service")
    @with_timeout("causal-analysis-service")
    async def generate_counterfactual_scenario(self, 
                                              data: Dict[str, Any], 
                                              intervention: Dict[str, Any],
                                              config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a counterfactual scenario based on the intervention.

        Args:
            data: The data to analyze
            intervention: The intervention to apply
            config: Optional configuration parameters

        Returns:
            A dictionary containing the counterfactual scenario
        """
        url = f"{self.base_url}/api/v1/counterfactual-scenario"
        payload = {
            "data": data,
            "intervention": intervention,
            "config": config or {}
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("causal-analysis-service")
    @with_retry("causal-analysis-service")
    @with_timeout("causal-analysis-service")
    async def analyze_currency_pair_relationships(self,
                                                price_data: Dict[str, Dict[str, Any]],
                                                max_lag: Optional[int] = 5,
                                                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Discovers causal relationships between currency pairs.
        
        Uses Granger causality to identify which currency pairs lead or cause movements in others.

        Args:
            price_data: Dictionary of price data for each currency pair
            max_lag: Maximum lag to consider for Granger causality
            config: Optional configuration parameters

        Returns:
            A dictionary containing the currency pair relationship analysis
        """
        url = f"{self.base_url}/api/v1/currency-pair-relationships"
        payload = {
            "price_data": price_data,
            "max_lag": max_lag,
            "config": config or {}
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("causal-analysis-service")
    @with_retry("causal-analysis-service")
    @with_timeout("causal-analysis-service")
    async def analyze_regime_change_drivers(self,
                                          market_data: Dict[str, Any],
                                          regime_column: str,
                                          feature_columns: List[str],
                                          config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Discovers causal factors that drive market regime changes.
        
        Identifies which features have the strongest causal influence on regime transitions.

        Args:
            market_data: Market data containing regime information and features
            regime_column: Column name for the regime information
            feature_columns: List of feature column names to analyze
            config: Optional configuration parameters

        Returns:
            A dictionary containing the regime change driver analysis
        """
        url = f"{self.base_url}/api/v1/regime-change-drivers"
        payload = {
            "market_data": market_data,
            "regime_column": regime_column,
            "feature_columns": feature_columns,
            "config": config or {}
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("causal-analysis-service")
    @with_retry("causal-analysis-service")
    @with_timeout("causal-analysis-service")
    async def enhance_trading_signals(self,
                                     signals: List[Dict[str, Any]],
                                     market_data: Dict[str, Any],
                                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhances trading signals with causal insights.
        
        Adds confidence adjustments, explanatory factors, conflicting signals,
        and expected duration based on causal analysis.

        Args:
            signals: List of trading signals to enhance
            market_data: Market data for context
            config: Optional configuration parameters

        Returns:
            A dictionary containing the enhanced trading signals
        """
        url = f"{self.base_url}/api/v1/enhance-trading-signals"
        payload = {
            "signals": signals,
            "market_data": market_data,
            "config": config or {}
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("causal-analysis-service")
    @with_retry("causal-analysis-service")
    @with_timeout("causal-analysis-service")
    async def assess_correlation_breakdown_risk(self,
                                              correlation_data: Dict[str, Any],
                                              config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Uses causal models to assess correlation breakdown risk between assets.
        
        Identifies pairs at risk of correlation breakdown and potential triggers.

        Args:
            correlation_data: Correlation data between assets
            config: Optional configuration parameters

        Returns:
            A dictionary containing the correlation breakdown risk assessment
        """
        url = f"{self.base_url}/api/v1/correlation-breakdown-risk"
        payload = {
            "correlation_data": correlation_data,
            "config": config or {}
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()