"""
Causal Analysis Models

This module defines the data models for causal analysis requests and responses.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class CausalGraphRequest(BaseModel):
    """
    Request model for generating a causal graph.
    """
    symbol: str = Field(..., description="The currency pair or symbol to analyze")
    timeframe: str = Field(..., description="The timeframe for the data (e.g., '1h', '4h', '1d')")
    start_date: datetime = Field(..., description="The start date for the analysis")
    end_date: Optional[datetime] = Field(None, description="The end date for the analysis")
    variables: Optional[List[str]] = Field(None, description="List of variables to include in the analysis")
    algorithm: str = Field("granger", description="Causal discovery algorithm to use (granger, pc, dowhy)")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for the algorithm")


class Edge(BaseModel):
    """
    Edge in a causal graph.
    """
    source: str = Field(..., description="Source node")
    target: str = Field(..., description="Target node")
    weight: float = Field(..., description="Edge weight")


class CausalGraphResponse(BaseModel):
    """
    Response model for a causal graph.
    """
    graph_id: str = Field(..., description="Unique identifier for the graph")
    nodes: List[str] = Field(..., description="List of nodes in the graph")
    edges: List[Edge] = Field(..., description="List of edges in the graph")
    adjacency_matrix: Optional[List[List[float]]] = Field(None, description="Adjacency matrix representation")
    created_at: datetime = Field(..., description="Creation timestamp")
    algorithm: str = Field(..., description="Algorithm used for causal discovery")
    parameters: Dict[str, Any] = Field(..., description="Parameters used for causal discovery")


class InterventionEffectRequest(BaseModel):
    """
    Request model for analyzing intervention effects.
    """
    symbol: str = Field(..., description="The currency pair or symbol to analyze")
    timeframe: str = Field(..., description="The timeframe for the data (e.g., '1h', '4h', '1d')")
    start_date: datetime = Field(..., description="The start date for the analysis")
    end_date: Optional[datetime] = Field(None, description="The end date for the analysis")
    treatment: str = Field(..., description="The treatment variable")
    outcome: str = Field(..., description="The outcome variable")
    confounders: Optional[List[str]] = Field(None, description="List of confounding variables")
    algorithm: str = Field("dowhy", description="Causal effect estimation algorithm to use")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for the algorithm")


class InterventionEffectResponse(BaseModel):
    """
    Response model for intervention effect analysis.
    """
    effect_id: str = Field(..., description="Unique identifier for the effect analysis")
    treatment: str = Field(..., description="The treatment variable")
    outcome: str = Field(..., description="The outcome variable")
    causal_effect: float = Field(..., description="Estimated causal effect")
    confidence_interval: Optional[List[float]] = Field(None, description="Confidence interval for the effect")
    p_value: Optional[float] = Field(None, description="P-value for the effect")
    created_at: datetime = Field(..., description="Creation timestamp")
    algorithm: str = Field(..., description="Algorithm used for effect estimation")
    parameters: Dict[str, Any] = Field(..., description="Parameters used for effect estimation")
    refutation_results: Optional[List[Dict[str, Any]]] = Field(None, description="Results of refutation tests")


class CounterfactualRequest(BaseModel):
    """
    Request model for generating counterfactual scenarios.
    """
    symbol: str = Field(..., description="The currency pair or symbol to analyze")
    timeframe: str = Field(..., description="The timeframe for the data (e.g., '1h', '4h', '1d')")
    start_date: datetime = Field(..., description="The start date for the analysis")
    end_date: Optional[datetime] = Field(None, description="The end date for the analysis")
    intervention: Dict[str, Union[float, int, str]] = Field(..., description="Intervention values for variables")
    target_variables: List[str] = Field(..., description="Variables to predict counterfactual values for")
    algorithm: str = Field("counterfactual", description="Counterfactual generation algorithm to use")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for the algorithm")


class CounterfactualResponse(BaseModel):
    """
    Response model for counterfactual scenario generation.
    """
    counterfactual_id: str = Field(..., description="Unique identifier for the counterfactual scenario")
    intervention: Dict[str, Union[float, int, str]] = Field(..., description="Intervention values for variables")
    target_variables: List[str] = Field(..., description="Variables with predicted counterfactual values")
    counterfactual_values: Dict[str, List[float]] = Field(..., description="Predicted counterfactual values")
    created_at: datetime = Field(..., description="Creation timestamp")
    algorithm: str = Field(..., description="Algorithm used for counterfactual generation")
    parameters: Dict[str, Any] = Field(..., description="Parameters used for counterfactual generation")


class CurrencyPairRelationshipRequest(BaseModel):
    """
    Request model for discovering causal relationships between currency pairs.
    """
    symbols: List[str] = Field(..., description="List of currency pairs to analyze")
    timeframe: str = Field(..., description="The timeframe for the data (e.g., '1h', '4h', '1d')")
    start_date: datetime = Field(..., description="The start date for the analysis")
    end_date: Optional[datetime] = Field(None, description="The end date for the analysis")
    variables: Optional[List[str]] = Field(None, description="List of variables to include in the analysis")
    algorithm: str = Field("granger", description="Causal discovery algorithm to use (granger, pc, dowhy)")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for the algorithm")


class CurrencyPairRelationshipResponse(BaseModel):
    """
    Response model for currency pair relationship analysis.
    """
    relationship_id: str = Field(..., description="Unique identifier for the relationship analysis")
    symbols: List[str] = Field(..., description="List of currency pairs analyzed")
    nodes: List[str] = Field(..., description="List of nodes in the graph")
    edges: List[Edge] = Field(..., description="List of edges in the graph")
    created_at: datetime = Field(..., description="Creation timestamp")
    algorithm: str = Field(..., description="Algorithm used for causal discovery")
    parameters: Dict[str, Any] = Field(..., description="Parameters used for causal discovery")


class RegimeChangeDriverRequest(BaseModel):
    """
    Request model for discovering causal factors that drive market regime changes.
    """
    symbol: str = Field(..., description="The currency pair or symbol to analyze")
    timeframe: str = Field(..., description="The timeframe for the data (e.g., '1h', '4h', '1d')")
    start_date: datetime = Field(..., description="The start date for the analysis")
    end_date: Optional[datetime] = Field(None, description="The end date for the analysis")
    regime_variable: str = Field("regime", description="Variable representing the market regime")
    potential_drivers: Optional[List[str]] = Field(None, description="List of potential driving variables")
    algorithm: str = Field("dowhy", description="Causal discovery algorithm to use")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for the algorithm")


class RegimeChangeDriverResponse(BaseModel):
    """
    Response model for regime change driver analysis.
    """
    driver_id: str = Field(..., description="Unique identifier for the driver analysis")
    regime_variable: str = Field(..., description="Variable representing the market regime")
    drivers: List[Dict[str, Any]] = Field(..., description="List of identified drivers with effect sizes")
    created_at: datetime = Field(..., description="Creation timestamp")
    algorithm: str = Field(..., description="Algorithm used for causal discovery")
    parameters: Dict[str, Any] = Field(..., description="Parameters used for causal discovery")


class TradingSignalEnhancementRequest(BaseModel):
    """
    Request model for enhancing trading signals with causal insights.
    """
    signals: List[Dict[str, Any]] = Field(..., description="List of trading signals to enhance")
    market_data: Dict[str, List[float]] = Field(..., description="Market data for causal analysis")
    config: Optional[Dict[str, Any]] = Field(None, description="Configuration for signal enhancement")


class TradingSignalEnhancementResponse(BaseModel):
    """
    Response model for trading signal enhancement.
    """
    enhanced_signals: List[Dict[str, Any]] = Field(..., description="List of enhanced trading signals")
    count: int = Field(..., description="Number of enhanced signals")
    causal_factors_considered: List[str] = Field(..., description="List of causal factors considered")


class CorrelationBreakdownRiskRequest(BaseModel):
    """
    Request model for assessing correlation breakdown risk between assets.
    """
    symbols: List[str] = Field(..., description="List of symbols to analyze")
    timeframe: str = Field(..., description="The timeframe for the data (e.g., '1h', '4h', '1d')")
    start_date: datetime = Field(..., description="The start date for the analysis")
    end_date: Optional[datetime] = Field(None, description="The end date for the analysis")
    stress_scenarios: Optional[List[Dict[str, Any]]] = Field(None, description="List of stress scenarios to test")
    algorithm: str = Field("counterfactual", description="Algorithm to use for risk assessment")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for the algorithm")


class CorrelationBreakdownRiskResponse(BaseModel):
    """
    Response model for correlation breakdown risk assessment.
    """
    risk_id: str = Field(..., description="Unique identifier for the risk assessment")
    symbols: List[str] = Field(..., description="List of symbols analyzed")
    baseline_correlations: Dict[str, Dict[str, float]] = Field(..., description="Baseline correlation matrix")
    stress_correlations: Dict[str, Dict[str, Dict[str, float]]] = Field(..., description="Correlation matrices under stress")
    breakdown_risk_scores: Dict[str, Dict[str, float]] = Field(..., description="Risk scores for correlation breakdown")
    created_at: datetime = Field(..., description="Creation timestamp")
    algorithm: str = Field(..., description="Algorithm used for risk assessment")
    parameters: Dict[str, Any] = Field(..., description="Parameters used for risk assessment")