"""
Interface for the Causal Analysis Service.

This module defines the interface for the Causal Analysis Service, which provides
causal analysis capabilities, including causal graph generation, intervention effect
analysis, and counterfactual scenario generation.
"""
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


class ICausalAnalysisService(ABC):
    """Interface for the Causal Analysis Service."""

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass