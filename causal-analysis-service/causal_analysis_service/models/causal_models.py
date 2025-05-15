"""
Causal models for the causal analysis service.
"""
from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class CausalAlgorithm(str, Enum):
    """
    Causal discovery algorithms.
    """
    PC = "pc"
    GRANGER = "granger"
    DOWHY = "dowhy"
    COUNTERFACTUAL = "counterfactual"

class CausalGraphRequest(BaseModel):
    """
    Request model for generating a causal graph.
    """
    data: Optional[Dict[str, List[float]]] = Field(None, description="Data for causal discovery")
    algorithm: CausalAlgorithm = Field(CausalAlgorithm.PC, description="Causal discovery algorithm")
    significance_level: float = Field(0.05, description="Significance level for statistical tests")
    max_lag: int = Field(1, description="Maximum lag for time series analysis")
    
    # Additional fields for test_causal_service.py
    symbol: Optional[str] = Field(None, description="Symbol for causal discovery")
    timeframe: Optional[str] = Field(None, description="Timeframe for causal discovery")
    start_date: Optional[datetime] = Field(None, description="Start date for causal discovery")
    end_date: Optional[datetime] = Field(None, description="End date for causal discovery")

class CausalGraphResponse(BaseModel):
    """
    Response model for a causal graph.
    """
    graph_id: Optional[str] = Field(None, description="Unique identifier for the graph")
    graph: Dict[str, List[str]] = Field(..., description="Causal graph as adjacency list")
    edge_weights: Dict[str, Dict[str, float]] = Field(..., description="Edge weights")
    node_metadata: Dict[str, Dict[str, Any]] = Field(..., description="Node metadata")
    algorithm_used: CausalAlgorithm = Field(..., description="Algorithm used for causal discovery")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    timestamp: datetime = Field(..., description="Timestamp of graph creation")

class InterventionEffectRequest(BaseModel):
    """
    Request model for calculating intervention effects.
    """
    graph_id: Optional[str] = Field(None, description="ID of the causal graph")
    intervention_variable: Optional[str] = Field(None, description="Variable to intervene on")
    intervention_value: Optional[float] = Field(None, description="Value to set the intervention variable to")
    target_variables: Optional[List[str]] = Field(None, description="Variables to calculate effects for")
    
    # Additional fields for test_causal_service.py
    symbol: Optional[str] = Field(None, description="Symbol for intervention effect analysis")
    timeframe: Optional[str] = Field(None, description="Timeframe for intervention effect analysis")
    start_date: Optional[datetime] = Field(None, description="Start date for intervention effect analysis")
    end_date: Optional[datetime] = Field(None, description="End date for intervention effect analysis")
    treatment: Optional[str] = Field(None, description="Treatment variable")
    outcome: Optional[str] = Field(None, description="Outcome variable")
    algorithm: Optional[str] = Field(None, description="Algorithm for intervention effect analysis")

class InterventionEffectResponse(BaseModel):
    """
    Response model for intervention effects.
    """
    effects: Dict[str, float] = Field(..., description="Calculated effects on target variables")
    confidence_intervals: Dict[str, List[float]] = Field(..., description="Confidence intervals for effects")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    timestamp: datetime = Field(..., description="Timestamp of calculation")

class CounterfactualScenarioRequest(BaseModel):
    """
    Request model for generating counterfactual scenarios.
    """
    graph_id: Optional[str] = Field(None, description="ID of the causal graph")
    factual_values: Optional[Dict[str, float]] = Field(None, description="Factual values of variables")
    intervention_variables: Optional[Dict[str, float]] = Field(None, description="Variables to intervene on and their values")
    target_variables: Optional[List[str]] = Field(None, description="Variables to calculate counterfactuals for")

class CounterfactualScenarioResponse(BaseModel):
    """
    Response model for counterfactual scenarios.
    """
    counterfactual_values: Dict[str, float] = Field(..., description="Counterfactual values of target variables")
    confidence_intervals: Dict[str, List[float]] = Field(..., description="Confidence intervals for counterfactuals")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    timestamp: datetime = Field(..., description="Timestamp of calculation")

# Additional models for test_causal_service.py

class CounterfactualRequest(BaseModel):
    """
    Request model for generating counterfactual scenarios.
    """
    symbol: str = Field(..., description="Symbol for counterfactual analysis")
    timeframe: str = Field(..., description="Timeframe for counterfactual analysis")
    start_date: datetime = Field(..., description="Start date for counterfactual analysis")
    end_date: datetime = Field(..., description="End date for counterfactual analysis")
    intervention: Dict[str, float] = Field(..., description="Intervention variables and values")
    target_variables: List[str] = Field(..., description="Variables to calculate counterfactuals for")
    algorithm: str = Field(..., description="Algorithm for counterfactual analysis")

class CurrencyPairRelationshipRequest(BaseModel):
    """
    Request model for discovering currency pair relationships.
    """
    symbols: List[str] = Field(..., description="Symbols for relationship analysis")
    timeframe: str = Field(..., description="Timeframe for relationship analysis")
    start_date: datetime = Field(..., description="Start date for relationship analysis")
    end_date: datetime = Field(..., description="End date for relationship analysis")
    algorithm: str = Field(..., description="Algorithm for relationship analysis")

class RegimeChangeDriverRequest(BaseModel):
    """
    Request model for discovering regime change drivers.
    """
    symbol: str = Field(..., description="Symbol for regime change analysis")
    timeframe: str = Field(..., description="Timeframe for regime change analysis")
    start_date: datetime = Field(..., description="Start date for regime change analysis")
    end_date: datetime = Field(..., description="End date for regime change analysis")
    regime_variable: str = Field(..., description="Regime variable")
    algorithm: str = Field(..., description="Algorithm for regime change analysis")

class TradingSignalEnhancementRequest(BaseModel):
    """
    Request model for enhancing trading signals.
    """
    signals: List[Dict[str, Any]] = Field(..., description="Trading signals to enhance")
    market_data: Dict[str, List[float]] = Field(..., description="Market data for enhancement")

class CorrelationBreakdownRiskRequest(BaseModel):
    """
    Request model for assessing correlation breakdown risk.
    """
    symbols: List[str] = Field(..., description="Symbols for correlation analysis")
    timeframe: str = Field(..., description="Timeframe for correlation analysis")
    start_date: datetime = Field(..., description="Start date for correlation analysis")
    end_date: datetime = Field(..., description="End date for correlation analysis")
    stress_scenarios: List[Dict[str, Any]] = Field(..., description="Stress scenarios")
    algorithm: str = Field(..., description="Algorithm for correlation analysis")